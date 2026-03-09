"Transformer"
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import math
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from typing import Optional, List, Tuple, Union, Callable, Mapping
from omegaconf import OmegaConf
from functools import partial


# internal 
from .. import misc
from ..torch_utils import masked_pooling
from .attention import DenseNetwork, MultiHeadAttention, repad_tensor

########### Layers ###########

class TransformerEncoderLayer(nn.Module):
    """self attention follow 'attention is all you need' and
    https://arxiv.org/pdf/2002.04745v1.pdf
    """
    def __init__(
        self,
        model_dim: int,
        mha_config: Union[Mapping,None] = None,
        dense_cfg: Union[Mapping,None] = None,
        ctxt_dim: int = 0,
        FiLM_dim: int = 0,
        topk: float = None,
        ) -> None:
        
        super().__init__()
        mha_config = mha_config or {}
        dense_cfg = dense_cfg or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim
        self.FiLM_dim=FiLM_dim
        self.topk = topk

        self.concat_bool = self.ctxt_dim>0

        if (self.topk is not None):
            raise NotImplementedError("Unpadded and topk cannot be used together atm")

        # dense network has to have init norm
        dense_cfg["init_norm"] = True if self.concat_bool else False
        self.dense_cfg=dense_cfg

        # The basic blocks
        self.self_attn = MultiHeadAttention(model_dim, model_dim, **mha_config)

        # Initial MLP
        self.dense = DenseNetwork(model_dim, model_dim, ctxt_dim=ctxt_dim if self.concat_bool else None, **dense_cfg)
        
        # The pre MHA and pre FFN layer normalisations
        self.layernorm_MLP = None
        if not self.concat_bool:
            self.layernorm_MLP = nn.LayerNorm(model_dim)

        self.layernorm_MHA = nn.LayerNorm(model_dim)

        if self.FiLM_dim>0:
            # Initial film layer
            self.FiLM_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.FiLM_dim, 6 * self.model_dim)
        )
            
            # Following film paper - set the weights/bias to 0
            # predict scale is (1+gamma)
            with T.no_grad():
                self.FiLM_layer[-1].weight.fill_(0)
                self.FiLM_layer[-1].bias.fill_(0)

    
    def FiLM(self, ctxt: T.Tensor,  mask_vk: Optional[T.BoolTensor] = None) -> T.Tensor:
        """Apply FiLM to the input tensor."""
        # calculate FiLM values
        
        if ctxt is None:
            return 1, 1, 0, 0, 0, 0
        
        # they used silu in the original paper
        FiLM_vals = self.FiLM_layer(ctxt)
        
        # repeat FiLM values to the batch size if using flashattn2
        if mask_vk.dtype == T.int64:
            FiLM_vals = T.repeat_interleave(FiLM_vals, mask_vk, dim=0)
        else:
            # input a PC
            FiLM_vals = FiLM_vals.unsqueeze(1)

        return T.chunk(FiLM_vals, 6, -1)

    def encoder(
        self,
        x: T.Tensor,
        mask_vk: Optional[T.BoolTensor] = None,
        ctxt: Union[T.Tensor,None] = None,
        FiLM: Union[T.Tensor,None] = None,
        attn_mask: Optional[T.BoolTensor] = None,
        unpad: bool = False,
    ) -> T.Tensor:
        """Pass using residual connections and layer normalisation."""
        
        # perform FiLM if ctxt is given
        a0,a1,s0,b0,s1,b1 = self.FiLM(FiLM, mask_vk=mask_vk)

        # self attention
        x = x + a0*self.self_attn(
            (1+s0)*self.layernorm_MHA(x)+b0,mask_vk=mask_vk,
            mask_q=mask_vk, attn_mask=attn_mask,
            unpad=unpad
            )
           
        # clone for skip connection
        x_mlp = x.clone()
        
        # layernorm
        if self.layernorm_MLP is not None:
            x_mlp = self.layernorm_MLP(x_mlp)

        # MLP(FiLM(x))+skip connection
        x = x + a1*self.dense((1+s1)*x_mlp + b1,
                              ctxt=ctxt if self.concat_bool else None,
                              cu_seqlens=mask_vk if mask_vk.dtype is not T.bool else None)

        return x

    def forward(
        self,
        x: T.Tensor,
        mask_vk: Optional[T.BoolTensor] = None,
        ctxt: Union[T.Tensor,None] = None,
        FiLM: Union[T.Tensor,None] = None,
        # attn_bias: Union[T.Tensor,None] = None,
        attn_mask: Optional[T.BoolTensor] = None,
        unpad: bool=False,
    ) -> T.Tensor:
        """Pass using residual connections and layer normalisation."""

        x = self.encoder(x, mask_vk, ctxt=ctxt, attn_mask=attn_mask, unpad=unpad,
                         FiLM=FiLM)

        return x

class TransformerDecoderLayer(nn.Module):
    """A transformer dencoder layer based on the GPT-2+Normformer style arcitecture.

    It contains:
    - self-attention-block
    - cross-attention block
    - dense network

    Layer norm is applied before each layer
    Residual connections are used, bypassing each layer

    Attention masks and biases are only applied to the self attention operation
    """

    def __init__(
        self,
        model_dim: int,
        mha_config: Union[Mapping,None] = None,
        dense_cfg: Union[Mapping,None] = None,
        ctxt_dim: int = 0,
        init_self_attn: bool=False,
        input_dim: int=None
    ) -> None:
        """
        Args:
            mha_config: Keyword arguments for multiheaded-attention block
            dense_cfg: Keyword arguments for feed forward network
        """
        super().__init__()
        mha_config = mha_config or {}
        dense_cfg = dense_cfg or {}
        self.model_dim = model_dim
        self.input_dim=input_dim
        if self.input_dim is None:
            self.input_dim=self.model_dim
        self.ctxt_dim = ctxt_dim
        self.init_self_attn = init_self_attn

        # The basic blocks
        if self.init_self_attn:
            self.self_attn = MultiHeadAttention(
                self.input_dim, self.input_dim, **mha_config
            )
            self.norm_preSA = nn.LayerNorm(self.input_dim)

        self.cross_attn = MultiHeadAttention(
            self.input_dim, self.input_dim, **mha_config
        )
        
        # Ensure there is layernorm in MLP
        # dense_cfg.update({"norm": "layer"})
        
        # MLP
        self.dense = DenseNetwork(
            self.input_dim, model_dim, ctxt_dim=ctxt_dim, **dense_cfg
        )

        # The pre_operation normalisation layers (lots from Foundation Transformers)
        self.norm_preC1 = nn.LayerNorm(self.input_dim)
        self.norm_preC2 = nn.LayerNorm(self.input_dim)
        # self.norm_preNN = nn.LayerNorm(self.input_dim)

    def forward(
        self,
        q_seq: T.Tensor,
        kv_seq: T.Tensor,
        mask_q: Optional[T.BoolTensor] = None,
        mask_vk: Optional[T.BoolTensor] = None,
        ctxt: Union[T.Tensor,None] = None,
        attn_bias: Union[T.Tensor,None] = None,
        attn_mask: Optional[T.BoolTensor] = None,
        unpad: Optional[T.BoolTensor] = False,
        **kwargs,
    ) -> T.Tensor:
        """Pass using residual connections and layer normalisation."""

        # Apply the self attention residual update
        if self.init_self_attn:
            q_seq = q_seq + self.self_attn(
                self.norm_preSA(q_seq),
                mask_vk=mask_q,
                attn_mask=attn_mask,
                # attn_bias=attn_bias,
                unpad=unpad,
            )

        # Apply the cross attention residual update
        # Note mask_q is only important for the self attention
        q_seq = q_seq + self.cross_attn(
            q=self.norm_preC1(q_seq), k=self.norm_preC2(kv_seq), mask_vk=mask_vk,
            mask_q=mask_q,
            unpad=unpad,
        )

        # Apply the dense residual update
        q_seq = q_seq + self.dense(
            q_seq, ctxt,
            cu_seqlens=mask_q if unpad else None)
            # cu_seqlens=mask_q if (mask_q is not None and mask_q.dtype is not T.bool) else None)

        return q_seq

class PerceiverLayer(nn.Module):
    def __init__(self, latent_dims:list, encode_cfg:dict, decode_cfg:dict=None,
                 process_cfg:dict=None, dense_cfg:dict=None,
                 n_processes:int=0, device:str="cuda"):
        super().__init__()
        self.latent_dims=latent_dims
        self.process_cfg=process_cfg
        self.n_processes=n_processes
        self.encode_cfg=encode_cfg
        self.decode_cfg=decode_cfg
        self.dense_cfg=dense_cfg if dense_cfg!=None else {}
        self.device = device
        
        if 'model_dim' not in self.encode_cfg:
            self.encode_cfg['model_dim'] = self.latent_dims[-1]

        # Layers
        self.processing_layers = nn.ModuleList([])

        #init network
        self.get_network()
        
        self.to(self.device)

    def get_network(self) -> None:

        # trainable latent space
        self.latent_arr = T.nn.Parameter(T.randn(*self.latent_dims))
        
        ### Encoder
        self.encode_layer = TransformerDecoder(**self.encode_cfg)
        
        ### Processor
        if self.process_cfg is not None:
            for _ in range(self.n_processes):
                self.processing_layers.append(
                    TransformerEncoder(**self.process_cfg))
        
        ### decoder
        if self.decode_cfg is not None:
            self.decode_layer = TransformerDecoder(**self.decode_cfg)

    def forward(self, input_ten:T.Tensor, ctxt_ten:T.Tensor, mask_vk:T.Tensor,
                ctxt:T.Tensor=None) -> T.Tensor:

        # expanding the latent tensor
        latent_ten = self.latent_arr.expand(len(ctxt_ten),*self.latent_arr.shape)
        
        ### Encode ctxt_ten to latent_ten
        latent_ten = self.encode_layer(latent_ten, ctxt_ten, mask_vk=mask_vk, ctxt=ctxt)

        ### Processor
        for nr_layer in range(self.n_processes):
            latent_ten = self.processing_layers[nr_layer](latent_ten, ctxt=ctxt)

        if self.decode_cfg is not None:
            ### Decoder latent_ten to input_ten
            return self.decode_layer(input_ten, latent_ten, ctxt=ctxt)
        else:
            return latent_ten


########### Vamila blocks ###########

class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers followed by a final normalisation step.

    Sequence -> Sequence
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_layers: int = 3,
        mha_config: Union[Mapping,None] = None,
        dense_cfg: Union[Mapping,None] = None,
        ctxt_dim: int = 0,
        FiLM_dim:int=0,
        n_registers:int=0,
        topk: float = None,
        out_norm:bool=False,
        device="cpu",
        # **kwargs,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_cfg: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.out_norm=out_norm
        self.n_registers=n_registers
        self.ctxt_dim=ctxt_dim
        self.FiLM_dim=FiLM_dim

        self.__out_registers=None
        self.device=device

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(self.model_dim, mha_config, dense_cfg, ctxt_dim=self.ctxt_dim,
                                        topk=topk,
                                        FiLM_dim=self.FiLM_dim)
                for _ in range(num_layers)
            ]
        )

        if self.out_norm:
            self.final_norm = nn.LayerNorm(model_dim)

        # init registers
        if self.n_registers>0:
            self.__registers = nn.Parameter(T.randn(n_registers, model_dim))

        self.to(self.device)

    def insert_registers(self, x: T.Tensor, registers:T.Tensor=None,
                              register_already_there:bool=False) -> T.Tensor:
        """Insert the registers to the input tensor or wrap out registers with new."""
        # expand registers to batch size
            
        if (registers is not None):
            #add dimension to axis 1
            if len(registers.shape)!=len(x.shape):
                registers = registers.view(*registers.shape[:1], 1, *registers.shape[1:])
        else:
            registers = self.__registers.expand(x.shape[0], *self.__registers.shape)
        
        # concat registers to input tensor
        if not register_already_there:
            x = T.cat([x, registers], -2)
        else:
            # remove n_registers and insert the new registers
            x = T.cat([x[:,:-self.n_registers ], registers], -2)

        return x

    def broadcast_registers(self, x: T.Tensor, mask:T.Tensor=None, registers:T.Tensor=None) -> T.Tensor:
        """Broadcast the registers/registers_mask to the input tensor."""
        
        # insert the registers
        x = self.insert_registers(x=x, registers=registers, register_already_there=False)
        
        # expand mask if provided
        if mask is not None:
            mask = T.cat([mask, T.ones(*mask.shape[:-1], self.n_registers, device=mask.device, dtype=bool)], -1)
        
        return x, mask
    
    def strip_registers(self, x: T.Tensor) -> T.Tensor:

        self.__out_registers = x[:,-self.n_registers: ]

        return x[:,:-self.n_registers ]
        
    def get_registers(self) -> T.Tensor:
        """Return the registers."""
        return self.__out_registers

    def get_latn_pc(self, mask_vk:T.Tensor=None) -> T.Tensor:
        """get PC with registers"""
        latn = self._latn_pc.clone()

        if self.n_registers>0:
            latn = T.concat([latn, self.get_registers()],1)
            
            if mask_vk is not None:
                # Determine the number of additional points
                additional_points = latn.shape[1] - mask_vk.shape[1]

                # Create a boolean tensor of True values for the additional points
                additional_mask = T.ones((mask_vk.shape[0], additional_points), dtype=T.bool, device=mask_vk.device)

                # Concatenate the additional_mask with the existing mask_vk along the second dimension
                mask_vk = T.cat([mask_vk, additional_mask], dim=1)

        return latn, mask_vk 
    
    def forward(self, x: T.Tensor, mask_vk:T.Tensor, ctxt_registers:T.Tensor=None, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        # 
        unpad=False
        if np.isin(x.dtype, [T.float16, T.bfloat16]):
            unpad = True
        
        # add registers
        if self.n_registers>0:
            x,mask_vk = self.broadcast_registers(x, mask=mask_vk, registers=ctxt_registers)
        
        # unpad after the ctxt_registers are added
        if unpad:
            original_mask_vk=mask_vk.clone()
            if len(x.shape) == 3:  # packed events
                x_dims = x.shape
                x = x[mask_vk]
                mask_vk = mask_vk.sum(-1)
        
        # run transformer layers
        for layer in self.layers:
            x = layer(x, mask_vk, unpad=unpad, **kwargs)
            
        # pad before returning with ctxt_registers etc
        if unpad and original_mask_vk.dtype is T.bool:
            x = repad_tensor(x, original_mask_vk, x_dims).contiguous()
            
        # remove ctxt_registers if added
        if self.n_registers>0:
            x = self.strip_registers(x)
        
        if self.out_norm:
            x = self.final_norm(x)

        self._latn_pc = x.clone()

        return x

class TransformerDecoder(nn.Module):
    """A stack of N transformer dencoder layers followed by a final normalisation step.

    Sequence x Sequence -> Sequence
    """

    def __init__(
        self,
        model_dim: int,
        num_layers: int = 1,
        mha_config: Union[Mapping,None] = None,
        dense_cfg: Union[Mapping,None] = None,
        ctxt_dim: int = 0,
        FiLM_dim: int = 0,
        init_self_attn:bool = False,
        input_dim: int=None,
        out_norm:bool=False,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_cfg: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context input
        """
        super().__init__()
        self.init_self_attn=init_self_attn
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.out_norm=out_norm
        if self.input_dim is None:
            self.input_dim=self.model_dim
        self.num_layers = num_layers
        self.ctxt_dim=ctxt_dim
        self.FiLM_dim=FiLM_dim
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(model_dim, mha_config, dense_cfg, ctxt_dim,
                                        init_self_attn=init_self_attn,
                                        input_dim=self.input_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm=None
        if self.out_norm:
            self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, q_seq: T.Tensor, kv_seq: T.Tensor,
                mask_vk:T.Tensor, mask_q:T.Tensor=None, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        unpad=False
        
        # for flashattn2
        if (np.isin(q_seq.dtype, [T.float16, T.bfloat16]) 
            and np.isin(kv_seq.dtype, [T.float16, T.bfloat16])):
            unpad = True
            original_mask_q = mask_q.clone()

            if len(q_seq.shape) == 3:  # packed events
                q_dims = q_seq.shape
                q_seq = q_seq[mask_q]
                mask_q = mask_q.sum(-1)

            if len(kv_seq.shape) == 3:  # packed events
                kv_seq = kv_seq[mask_vk]
                mask_vk = mask_vk.sum(-1)
        
        for layer in self.layers:
            q_seq = layer(q_seq, kv_seq, unpad=unpad, mask_q=mask_q, mask_vk=mask_vk, **kwargs)

        # pad before returning with ctxt_registers etc
        if unpad and original_mask_q.dtype is T.bool:
            q_seq = repad_tensor(q_seq, original_mask_q, q_dims).contiguous()

        if self.final_norm is not None:
            return self.final_norm(q_seq)
        else:
            return q_seq


########### Transformer architectures ###########

class ClassEmbedding(nn.Module):
    """A stack of N transformer encoder layers followed by a final normalisation step.

    Sequence -> Sequence
    """

    def __init__(self, cls_dim_in:int, encoder: partial, dense_cfg: partial,
                decoder: partial=None, cls_dim_out:int=None, d_model:int=None, **kwargs) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_cfg: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()

        self.cls_dim_in = cls_dim_in

        self.cls_dim_out = cls_dim_out if cls_dim_out is not None else cls_dim_in
        self.d_model = d_model if d_model is not None else cls_dim_in
        
        self.latent_scale=kwargs.get("latent_scale", 1)
        self.pooling = kwargs.get("pooling")
        self.kwargs = kwargs
            
        # networks
        self.dense_cfg=dense_cfg
        self.encoder=encoder
        self.decoder=decoder
        
        if self.pooling is not None and self.decoder is not None:
            raise ValueError("Pooling cannot be used with decoder or vice versa")

        self.get_network()

    def get_network(self) -> None:
        """Initialize the network"""
        if self.d_model != self.cls_dim_in:
            self.inpt_embed = nn.Linear(self.cls_dim_in, self.d_model)

        #encoder to encode input pc
        if self.encoder is not None:
            self.encoder = self.encoder(model_dim=self.d_model)

        if self.decoder is not None:# CaIT network

            # init global class token
            self.global_cls_token = nn.Parameter(
                T.randn((1, 1, self.d_model)) * self.latent_scale)

            # init decoder for pooling information from x to global_cls_token
            self.decoder = self.decoder(model_dim=self.d_model)

        elif self.encoder.n_registers==0: # if passed ViT network
            raise ValueError("No decoder or registers provided")
    
    def get_latn_pc(self, mask_vk:T.Tensor=None) -> T.Tensor:
        """get PC with registers"""
        latn = self._latn_pc.clone()

        if self.encoder.n_registers>0:
            latn = T.concat([latn, self.encoder.get_registers()],1)
            
            if mask_vk is not None:
                # Determine the number of additional points
                additional_points = latn.shape[1] - mask_vk.shape[1]

                # Create a boolean tensor of True values for the additional points
                additional_mask = T.ones((mask_vk.shape[0], additional_points), dtype=T.bool, device=mask_vk.device)

                # Concatenate the additional_mask with the existing mask_vk along the second dimension
                mask_vk = T.cat([mask_vk, additional_mask], dim=1)

        return latn, mask_vk 

    def forward(self, x: T.Tensor, mask:T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        
        if hasattr(self, "inpt_embed"):
            x = self.inpt_embed(x)
        
        # run encoders
        if self.encoder is not None:
            self._latn_pc = self.encoder(x, mask_vk=mask, **kwargs)
            
            # get pc + registers
            x, mask = self.get_latn_pc(mask_vk=mask)

        # run decoder and get last register as class token
        if self.pooling is not None:
            cls_token = masked_pooling(x, mask, pooling_style=self.pooling)
        elif self.decoder is not None: # CaiT network
            # expand cls token to batch size
            cls_token = self.global_cls_token.expand(
                x.shape[0], *self.global_cls_token.shape[1:]
                )

            # Going deeper with image transformer idea
            cls_token = self.decoder(cls_token, x, mask_vk=mask, **kwargs).squeeze(1)
        else:
            # ViT network - use the first register as cls token
            cls_token = self.encoder.get_registers()[:, 0, :]

        return cls_token

class Perceiver(nn.Module):
    """A stack of N transformer dencoder layers followed by a final normalisation step.

    Sequence x Sequence -> Sequence
    """

    def __init__(
        self,
        pcivr_cfg: Union[Mapping,None],
        num_layers: int = 1,
        device:str="cpu",
    ) -> None:
        """
        Args:
            pcivr_cfg: PerceiverLayer config
            num_layers: Number of encoder layers used
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PerceiverLayer(**pcivr_cfg)
                for _ in range(num_layers)
            ]
        )
        self.pcivr_cfg=pcivr_cfg
        self.num_layers=num_layers
        self.device=device

        self.to(self.device)
        # self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, input_ten: T.Tensor, ctxt_ten: T.Tensor=None, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        # if ctxt is not passed
        if ctxt_ten is None:
            ctxt_ten = input_ten.clone()
            
        # run model
        for layer in self.layers:
            input_ten = layer(input_ten, ctxt_ten, **kwargs)
        
        # output new tensor
        return input_ten
        # return self.final_norm(q_seq)

class UPerceiver(nn.Module):
    """A stack of N transformer dencoder layers followed by a final normalisation step.

    Sequence x Sequence -> Sequence
    """

    def __init__(
        self,
        input_dim: int,
        model_dims: List[int],
        cnts_sizes: List[int],
        pcivr_cfg: dict,
        ctxt_dim:int=0,
        FiLM_dim:int=0,
        device:str="cpu",
        **kwargs
    ) -> None:
        """
        Args:
            pcivr_cfg: PerceiverLayer config
            num_layers: Number of encoder layers used
        """
        super().__init__()
        self.input_dim=input_dim # [C,H, W]
        self.model_dims=model_dims
        self.ctxt_dim=ctxt_dim
        self.FiLM_dim=FiLM_dim
        self.all_model_dims = [self.input_dim]+self.model_dims
        self.cnts_sizes=cnts_sizes
        self.pcivr_cfg=pcivr_cfg
        self.device=device
        self.image_kwargs = kwargs.get("image_kwargs", None)

        self.down_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])

        self.down_nn_layers = nn.ModuleList([])
        self.up_nn_layers = nn.ModuleList([])
        
        self.get_network()
    
    def setup_for_images(self):
        if ('n_patches' and 'image_shape') not in self.image_kwargs:
            raise ValueError("Missing image_shape or n_patches in image_kwargs")

        self.image_shape = self.image_kwargs["image_shape"]
        self.n_patches = self.image_kwargs["n_patches"]

        if isinstance(self.image_shape, (int, np.int64)):
            self.image_shape = np.array([self.image_shape,self.image_shape])
        else:
            self.image_shape=np.array(self.image_shape)

        self.kernel_size = np.array(self.image_shape[1:])/(self.n_patches)
        
        if any(self.kernel_size!=np.int64(self.kernel_size)):
            raise ValueError("Image not divisible - change n_patches")
        self.kernel_size = np.int64(self.kernel_size)

        self.patch_features=np.int64(np.product(self.kernel_size))

        self.input_features = self.patch_features*self.image_shape[0]

        # image in the forward will be [B, H, W, C]
        self.pe = T.nn.Parameter(T.randn(*self.image_shape[::-1]))

        # fold/unfolding
        self.args = {"dilation":1, "padding":0, 
                        'kernel_size': tuple(self.kernel_size), 
                        'stride': self.kernel_size}

        self.unfold = nn.Unfold(**self.args)
        self.fold = nn.Fold(output_size=tuple(self.image_shape)[1:], **self.args)
        
    def get_network(self):

        if self.image_kwargs is not None:
            self.setup_for_images()
            
            # override number of input features
            self.input_dim = self.input_features

        # downscale TODO split into encoder and decoder
        self.down_nn_layers = nn.ModuleList(
            [nn.Linear(self.all_model_dims[i],self.all_model_dims[i+1])
             for i in range(len(self.all_model_dims)-1)]
            )

        self.up_nn_layers = nn.ModuleList(
            [nn.Linear(self.all_model_dims[::-1][i],self.all_model_dims[::-1][i+1])
             for i in range(len(self.all_model_dims)-1)]
            )

        for model_dim, cnts_size in zip(self.model_dims, self.cnts_sizes):
            encode_down_cfg = self.pcivr_cfg.copy()
            encode_down_cfg["latent_dims"] = (cnts_size, model_dim)
            encode_down_cfg["encode_cfg"]["model_dim"] = model_dim
            encode_down_cfg["encode_cfg"]["ctxt_dim"] = self.ctxt_dim
            encode_down_cfg["encode_cfg"]["FiLM_dim"] = self.FiLM_dim
            self.down_layers.append(PerceiverLayer(**encode_down_cfg))
            

        # upscale
        for model_dim, cnts_size in zip(self.model_dims[::-1][1:], self.cnts_sizes[::-1][1:]):
            decode_up_cfg = self.pcivr_cfg["encode_cfg"].copy()
            decode_up_cfg["model_dim"] = model_dim
            decode_up_cfg["ctxt_dim"] = self.ctxt_dim
            self.up_layers.append(TransformerDecoder(**decode_up_cfg))


        self.to(self.device)

    def forward(self, input_ten: T.Tensor, mask_vk: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        skip_features=[]
        
        if self.image_kwargs is not None:
            input_ten = input_ten+self.pe.expand_as(input_ten)
            # prepare image
            input_ten = self.unfold(input_ten.permute(0,-1,1,2)).permute(0,-1,1)



        for layer, nn_layer in zip(self.down_layers, self.down_nn_layers):
            skip_features.append(input_ten.clone())

            input_ten = nn_layer(input_ten)

            input_ten = layer(
                input_ten=None, ctxt_ten=input_ten,
                mask_vk=T.ones(*input_ten.shape[:-1], device=self.device)==1,
                **kwargs)

        # upscale pc with cross attention with skip connections
        for layer, nn_layer in zip(self.up_layers, self.up_nn_layers):
            skip_cnt = skip_features.pop()
            
            input_ten = nn_layer(input_ten)

            input_ten = layer(
                q_seq=skip_cnt, kv_seq=input_ten,
                mask_vk=T.ones(*input_ten.shape[:-1], device=self.device)==1,
                **kwargs
                )
            
        output = self.up_nn_layers[-1](input_ten)

        if self.image_kwargs is not None:
            # fold image back
            output = self.fold(output.permute(0,-1,1)).permute(0,2,3,1)

        return output

def test_get_data():
    config = misc.load_yaml("configs/data_cfgs/data_cfg.yaml")
    data = hydra.utils.instantiate(config.train_set)
    
    dataloader = hydra.utils.instantiate(config.loader_cfg)(data)
    dataloader = hydra.utils.instantiate(config.img_enc)(dataloader)
    #find std and mean of data
    data=[]
    data_ctxt=[]
    for nr, i in enumerate(dataloader):
        data.append(i[0] if len(i) else i)
        if i[1] is not None:
            data_ctxt.append(i[1] if len(i) else i)
        if nr==1:
            break
    data=T.concat(data)
    if data_ctxt[0] is not None:
        data_ctxt=T.concat(data_ctxt)
    return data

if __name__ == "__main__":
    import hydra
    dense_cfg = {"act_str": "gelu","n_layers": 0, "norm": "layer",  "nfactor": 2}
    
    if False: # testing flash attention

        # init random data
        pc_input = T.rand(512,20, 64, device="cuda").half()
        pc_mask = T.rand(512, 20, device="cuda") > 0.5
        
        # init transformer
        transformer = TransformerEncoder(model_dim=64, num_layers=3, mha_config={"n_heads": 4})
        
        out = transformer(pc_input, mask_vk=pc_mask)

    elif False: # test registers
        model_cfg = misc.load_yaml("/home/users/a/algren/work/bridge_diffusion/configs/model/architectures/class_embedding.yaml")
        # model_cfg.encoder.ctxt_dim=0
        transformer = hydra.utils.instantiate(model_cfg)()
        
        
        out = transformer(T.randn(128, 200, transformer.cls_dim_in), mask=T.randn(128, 200)>0.5)
        print(transformer.get_registers())
        print(transformer.__registers)
    else:
        pcivr_cfg = {
            'latent_dims': (128, 3),
            'encode_cfg':{
                "dense_cfg": {
                    'nfactor': 2,
                    'n_layers': 0,
                    'act_str': 'glu',
                    'dropout': 0.2},
                "num_layers": 4,
                'mha_config': 
                    {'attn_heads': 4},
                'out_norm': False,
            },
        }
        UNet = Perceiver(pcivr_cfg, 2)
        input_ten = T.randn(128, 128, 3)
        out = UNet(input_ten, mask_vk=input_ten.sum(-1)>1)
        
        model_cfg = {
        'input_dim': 3,
        'model_dims': [32, 64, 128],
        'cnts_sizes': [128, 64, 32],
        'pcivr_cfg': pcivr_cfg
        }
        UNet = UPerceiver(**model_cfg)
        
        input_ten = T.randn(128, 128, 3)
        output_ten = UNet(input_ten, mask_vk=input_ten.sum(-1)>1)
    



