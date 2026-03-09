"attention and misc"
from numpy import isin
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import math
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Callable, Mapping

# internal packages
from ..torch_utils import activation_functions
from ..modules import GatedLinearUnits


try:
    from flash_attn import (flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    # flash_attn_varlen_kvpacked_func
    )
    # from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func = None, None
    # flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None

class DenseNetwork(nn.Module):
    def __init__(self, in_features:int, out_features:int, n_layers:int=1,
                 ctxt_dim:int=0, act_str:str="leakyrelu", act_kwargs:dict=None,
                 zeroed=False, nfactor:int=1,hid_features:int=None,
                 dropout:float=0.0, init_norm:bool=False, norm:bool=True,
                 norm_type:str="layernorm",
                 ) -> None:
        super().__init__()
        self.zeroed=zeroed
        self.init_norm=init_norm
        self.layernorm= nn.LayerNorm if 'layer' in norm_type else nn.BatchNorm1d
        self.norm=norm
        self.in_features=in_features
        self.ctxt_dim = ctxt_dim if isinstance(ctxt_dim, (int, np.integer)) else 0
        self.in_features += self.ctxt_dim
        self.out_features=out_features
        self.hid_features=hid_features
        self.nfactor=nfactor
        self.dropout=dropout
        self.act_str=act_str
        self.act_kwargs={} if act_kwargs is None else act_kwargs


        if self.hid_features is None:
            self.hid_features=in_features*nfactor if in_features>out_features else out_features*nfactor

        # input layer scales up to hidn dim

        # for glu usually
        if 'glu' in self.act_str:
            self.layers = GatedLinearUnits(
                input_dims=self.in_features-self.ctxt_dim,
                hddn_dims=self.hid_features,
                out_dims = self.out_features,
                ctxt_dims=self.ctxt_dim,
                dropout=self.dropout,
                norm=self.norm or self.init_norm)
        else:
            self.layers = nn.Sequential()
            ### create network ###
            if self.init_norm:
                self.layers.extend([self.layernorm(self.in_features)])
            # add linear to scale up dims
            self.layers.extend(
                [
                nn.Linear(self.in_features, self.hid_features),
                activation_functions(act_str.casefold(), params=self.act_kwargs),
                nn.Dropout(self.dropout)
                ]
                )

            # loop that builds the network
            for _ in range(1, n_layers):

                # add norm
                if norm:
                    self.layers.append(self.layernorm(self.hid_features))

                # add linear
                self.layers.append(nn.Linear(self.hid_features, self.hid_features))
                
                # add act function
                self.layers.append(activation_functions(act_str.casefold(),
                                                        params=self.act_kwargs))

                # add dropout
                self.layers.append(nn.Dropout(self.dropout))

            # output layer
            self.layers.append(nn.Linear(self.hid_features, self.out_features))

            if self.zeroed:
                self.layers[-1].weight.data.fill_(0.00)
                self.layers[-1].bias.data.fill_(0.00)

    def forward(self, inpt: T.Tensor, ctxt: Union[T.Tensor,None] = None,
                **kwargs)->T.Tensor:

        cu_seqlens = kwargs.get("cu_seqlens")

        if ctxt is not None:
            # ensure ctxt has same dim as inpt
            if kwargs.get("cu_seqlens") is not None:
                ctxt = T.repeat_interleave(ctxt.to(inpt.dtype), cu_seqlens, dim=0)
            else:
                if len(inpt.shape) != len(ctxt.shape):
                    ctxt = ctxt.contiguous().view(*ctxt.shape[:1], 1, *ctxt.shape[1:])
                # if True: #inpt.shape[:-1] != ctxt.shape[:-1]:
                    ctxt = ctxt.expand(*inpt.shape[:-1], ctxt.shape[-1])
                
            # concat inpt/ctxt
            inpt = T.concat([inpt, ctxt], -1)
        
        # run MLP
        return self.layers(inpt)
    
def repad_tensor(tensor:T.Tensor, mask_vk:T.Tensor, dims: list):
    """Repad a tensor to the original size given the mask_vk"""
    padded_x = T.zeros(dims, device=tensor.device, dtype=tensor.dtype)
    padded_x[mask_vk] = tensor
    return padded_x

def merge_masks(
    # mask_q: Union[T.BoolTensor, None],
    mask_vk: Union[T.BoolTensor, None],
    # attn_mask: Union[T.BoolTensor, None],
    q_shape: T.Size,
    k_shape: T.Size,
    device: T.device,
) -> Union[None, T.BoolTensor]:
    """Create a full attention mask which incoporates the padding
    information."""

    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    # If either pad mask exists, create
    if mask_vk is not None:
    # if mask_q is not None or mask_vk is not None:
        # if mask_q is None:
        #     mask_q = T.full((q_shape[0], q_shape[1]), True, device=device)
        if mask_vk is None:
            mask_vk = T.full((k_shape[0], k_shape[1]), True, device=device)
        merged_mask = mask_vk.unsqueeze(-2).expand(-1, q_shape[-2], -1)
        # merged_mask = mask_q.unsqueeze(-1) & mask_vk.unsqueeze(-2)

    # If attention mask exists, create
    # if attn_mask is not None:
    #     merged_mask = attn_mask if merged_mask is None else attn_mask & merged_mask

    return merged_mask
    
def attention(query, key, value):
    "Compute 'Scaled Dot Product Attention'"
    # d_k is the number of features

    d_k = query.size(-1)

    scores = T.matmul(query, key.transpose(-2, -1).contiguous() ) / math.sqrt(d_k)

    p_attn = T.nn.functional.softmax(scores, dim = -1)

    return T.matmul(p_attn, value) #, p_attn


class MultiHeadAttention(nn.Module):
    """
    Simple mulit head attention WITHOUT skip connection at the end
    """ 
    def __init__(self, depth_q, depth_vk, attn_heads=4,
                 device="cpu", **kwargs):
        super().__init__()
        # self.depth=depth
        self.device=device
        self.attn_heads=attn_heads

        self.depth_q = int(depth_q)
        self.depth_vk = int(depth_vk)
        self.zero_init = kwargs.get("zero_init", False)

        # multi head blk
        if (
            (self.attn_heads>=self.depth_q) or
            (self.attn_heads>=self.depth_vk)
            ):
            print("Warning: attn_heads changed to 1")
            self.attn_heads=1

        self.depth_q_blk = self.depth_q//self.attn_heads
        self.depth_vk_blk = self.depth_vk//self.attn_heads

        if self.attn_heads*self.depth_vk_blk != self.depth_vk:
            raise ValueError("dimension not fitting")
        
        self.get_network()

    def get_network(self):
        ## init trainable modules
        self.attention_blks = nn.ModuleList([])

        # attention TODO merge into one
        self.W_query = nn.Linear(self.depth_q, self.depth_vk)

        self.W_key = nn.Linear(self.depth_vk, self.depth_vk)

        self.W_value = nn.Linear(self.depth_vk, self.depth_vk)
        
        self.out_proj = nn.Linear(self.depth_vk, self.depth_q)
            
        if self.zero_init:
            self.out_proj[-1].weight.data.fill_(0.00)
            self.out_proj[-1].bias.data.fill_(0.00)

        self.to(self.device)

    def classic_attention(self, q, k, v, attn_mask, batch_size):
        # Permute for the attention to apply each head independantly
        # B, num_heads, HxW, channels_per_head/features 
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Now we can apply the attention operation
        a_out = T.nn.functional.scaled_dot_product_attention(q, k, v,
                                                             attn_mask=attn_mask,
                                                             )

        # Concatenate the all of the heads together to get back to: B, model_dim/features, HxW
        a_out = a_out.transpose(1, -2).contiguous().view(batch_size, -1, self.depth_vk)

        # Pass through the final 1x1 convolution layer: B, q_dim, HxW
        return a_out
    
    def flash_attention(self, qkv, cu_seqlens_q, max_seq_len):
        out = flash_attn_varlen_qkvpacked_func(qkv,cu_seqlens_q,max_seq_len) # TODO maybe add dropout in attn
        b, n, m = out.shape
        return out.contiguous().view(b, n*m)

    def forward(self, q: T.Tensor, v: T.Tensor = None, k: T.Tensor = None,
                 mask_vk: T.Tensor=None, mask_q: T.Tensor=None,
                 attn_mask: T.Tensor=None, unpad:bool=False
                 ) -> T.Tensor:
        
        if (self.depth_q != self.depth_vk) & unpad:
            raise NotImplementedError("Unpad only working with self attention atm")
        

        # If only q is provided then we automatically apply self attention
        if k is None:
            k = q
        if v is None:
            v = k

        #attn mask
        if unpad:
            zero = T.zeros(1, dtype=mask_vk.dtype, device=mask_vk.device)
            cu_seqlens_vk = T.cat([zero, T.cumsum(mask_vk, dim=-1)]).to(T.int32)
            max_seq_len_vk = mask_vk.max()
            
            if q.shape[0]!=v.shape[0]:
                cu_seqlens_q = T.cat([zero, T.cumsum(mask_q, dim=-1)]).to(T.int32)
                max_seq_len_q = mask_q.max()
                
        elif (mask_vk is not None) or (mask_q is not None):
            attn_mask = merge_masks(mask_vk, q.shape, k.shape, q.device)
            # B x n_heads x L x S
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.attn_heads,1,1)

        # Project using the learnable layers: B, HxW/point-cloud, model_dim/features
        q = self.W_query(q)
        k = self.W_key(k)
        v = self.W_value(v)
        
        # Break (model_dim or features = num_heads x channels_per_head) for the different heads
        # B, HxW/point-cloud, num_heads, channels_per_head/features
        q = q.contiguous().view((len(q), -1, self.attn_heads, self.depth_q_blk))
        k = k.contiguous().view((len(k), -1, self.attn_heads, self.depth_vk_blk))
        v = v.contiguous().view((len(v), -1, self.attn_heads, self.depth_vk_blk))
        
        if q.shape[0]!=v.shape[0] and unpad:
            a_out = flash_attn_varlen_kvpacked_func(
                q=q.squeeze(1), cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seq_len_q, 
                kv=T.concat([k,v],1), cu_seqlens_k=cu_seqlens_vk, max_seqlen_k=max_seq_len_vk
                ).contiguous().view(-1, self.attn_heads*self.depth_q_blk)
        elif unpad:
            a_out = self.flash_attention(T.concat([q,k,v],1), cu_seqlens_vk,
                                          max_seq_len_vk)
        else:
            a_out = self.classic_attention(q, k, v, attn_mask, len(q))
        
        return self.out_proj(a_out)

if __name__ == "__main__":
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    pc_input = T.rand(512,20,3, device="cuda").half()
    pc_mask = T.rand(512, 20, device="cuda") > 0.5
    seq_len = pc_mask.sum(1).cpu()
    packed = pack_padded_sequence(pc_input, seq_len, batch_first=True, enforce_sorted=False)
    cu_seqlens_q = F.pad(seq_len.cumsum(0), pad=(1,0), value=0).to(T.int32)+1
    pc_input_unpad = pc_input[pc_mask]
    # Assuming `pc_input_unpad` is your tensor
    # Assuming `pc_input_unpad` is your tensor
    pc_input_unpad = pc_input_unpad.unsqueeze(1)  # This will change the shape to (b, 1, f)
    pc_input_unpad = pc_input_unpad.unsqueeze(1)  # This will change the shape to (b, 1, 1, f)

    # Now, we need to expand the size of the second dimension to 3
    pc_input_unpad = pc_input_unpad.expand(-1, 3, 1, -1)  # This will change the shape to (b, 3, 1, f)

    # Now, we need to expand the size of the second dimension to 3
    out = flash_attn_varlen_qkvpacked_func(pc_input_unpad,
                                     cu_seqlens_q,seq_len.max())