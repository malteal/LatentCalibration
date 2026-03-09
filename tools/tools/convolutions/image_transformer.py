"Transformer setup for images"
import math
from typing import Union
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from omegaconf import OmegaConf

# internal 
from tools import misc
from tools.transformers.attention import DenseNetwork
from tools.transformers.transformer import MultiHeadAttention
import tools.positional_encoding as pe

def positional_encoding(image_shape, pos_encode_kwargs, device):
    channels = image_shape[0]
    embedding = pe.sinusoidal(embedding_dims=channels,device="cpu",
                                **pos_encode_kwargs)
    h_index = T.linspace(0, 1, image_shape[-1])
    w_index = T.linspace(0, 1, image_shape[-2])
    h_embed = embedding(h_index[:, None])
    w_embed = embedding(w_index[:, None])
    pe_encoding = T.zeros(1, channels, image_shape[-2], image_shape[-1])

    for i in range(embedding.embedding_dims//2):
        pe_encoding[0, i:i+2, :, :] = T.stack(T.meshgrid(h_embed[:,i], w_embed[:,i],
                                                            indexing='xy'),0)
    return pe_encoding.to(device)

# class MultiHeadGateAttention(nn.Module):
#     # similar to https://arxiv.org/pdf/2103.06104.pdf
#     def __init__(self, depth_q, depth_vk, image_shape_q, image_shape_vk,
#                  pos_encode_kwargs, attn_heads=4, device="cuda", **kwargs):
#         if np.any(image_shape_q[0] != image_shape_vk[0]):
#             raise ValueError("Image dimension between q and vk has to be the same")
        
#         super().__init__(depth_q=depth_vk, depth_vk=depth_vk,
#                          image_shape_q=image_shape_q, image_shape_vk=image_shape_vk,
#                          pos_encode_kwargs=pos_encode_kwargs, attn_heads=attn_heads, device=device, **kwargs)
#         self.original_depth_q=depth_q
#         self.permute_layers = kwargs.get("permute_layers", None)
        
#         self.values_conv = nn.Sequential(nn.Conv2d(self.depth_vk, self.depth_vk, kernel_size=1),
#                                   nn.BatchNorm2d(self.depth_vk),
#                                   nn.SiLU())

#         self.keys_conv = nn.Sequential(nn.Conv2d(self.depth_vk, self.depth_vk, kernel_size=1),
#                                   nn.BatchNorm2d(self.depth_vk),
#                                   nn.SiLU())

#         self.queries_conv = nn.Sequential(nn.Conv2d(self.original_depth_q, self.depth_vk, kernel_size=1),
#                                   nn.BatchNorm2d(self.depth_vk),
#                                   nn.SiLU())

#         # Upsample before or after attention??!??!
#         #after results in every 2x2 is the same
#         self.conv = nn.Sequential(
#                                 # nn.Upsample(scale_factor=2, mode='nearest'),
#                                 nn.Conv2d(self.depth_vk, self.depth_vk, kernel_size=1),
#                                 # nn.BatchNorm2d(self.depth_vk),
#                                 nn.Sigmoid()
#                                   )
#         self.to(self.device)

#     def forward(self, values, keys, queries):

#         queries = self.queries_conv(queries)
#         values = self.values_conv(values)
#         keys = self.keys_conv(keys)

#         gate_images = self.image_forward(values, keys, queries)

#         return self.conv(gate_images) * values
    
class MultiHeadGateAttention(nn.Module):
    def __init__(self, image_shape_q, image_shape_vk, pos_encode_kwargs=None,
                 attn_heads=4, device="cuda", **kwargs):
        if np.any(image_shape_q[0] != image_shape_vk[0]):
            raise ValueError("Image dimension between q and vk has to be the same")
        super().__init__()
        self.depth_q=image_shape_q[-1]
        self.depth_vk=image_shape_vk[-1]
        self.image_shape_q=image_shape_q
        self.image_shape_vk=image_shape_vk
        self.pos_encode_kwargs={} if pos_encode_kwargs is None else pos_encode_kwargs
        self.attn_heads=attn_heads
        self.permute_layers = kwargs.get("permute_layers", None)
        self.trainable_pe=kwargs.get("trainable_pe", False)
        self.device=device
        
        # get network
        self.attention = MultiHeadAttention(self.depth_q, self.depth_vk,
                                            attn_heads=self.attn_heads,
                                            device=self.device, **kwargs)

        # positional encoding
        if (self.image_shape_q is not None):
            if self.trainable_pe:
                self.pos_encode_q =  T.nn.Parameter(T.randn(*self.image_shape_q))
                self.pos_encode_vk =  T.nn.Parameter(T.randn(*self.image_shape_vk))
            else:
                self.pos_encode_q = self.positional_encoding(self.image_shape_q,
                                                    self.pos_encode_kwargs,self.device)
                self.pos_encode_vk = self.positional_encoding(self.image_shape_vk,
                                                    self.pos_encode_kwargs,self.device)
        self.v_conv = nn.Sequential(nn.Conv2d(self.depth_vk, self.depth_vk, kernel_size=1),
                                  nn.BatchNorm2d(self.depth_vk),
                                  nn.SiLU())

        self.k_conv = nn.Sequential(nn.Conv2d(self.depth_vk, self.depth_vk, kernel_size=1),
                                  nn.BatchNorm2d(self.depth_vk),
                                  nn.SiLU())

        self.q_conv = nn.Sequential(nn.Conv2d(self.depth_q, self.depth_q, kernel_size=1),
                                  nn.BatchNorm2d(self.depth_q),
                                  nn.SiLU())

        # Upsample before or after attention??!??!
        #after results in every 2x2 is the same
        self.conv = nn.Sequential(
                                # nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(self.depth_q, self.depth_q, kernel_size=1),
                                nn.BatchNorm2d(self.depth_q),
                                nn.Sigmoid()
                                  )
        self.to(self.device)

    def forward(self, q:T.Tensor, k:T.Tensor, v:T.Tensor):
        # run through conv MLP
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        
        #permute layer for attention -  B, HxW/point-cloud, C/F
        q = self.permute_layers[0](q)
        k = self.permute_layers[0](k)
        v = self.permute_layers[0](v)
        
        # Add positional encoding
        q = q+self.pos_encode_q.expand_as(q)
        k = k+self.pos_encode_vk.expand_as(k)
        v = v+self.pos_encode_vk.expand_as(v)
        
        # Get the shape of the q tensor and save the original for later
        b, *spatial, c = q.shape

        # Flatten each image to combine the spacial dimensions: B, HxW/point-cloud, C/F
        q_flatten = T.flatten(q, 1, 2)
        k_flatten = T.flatten(k, 1, 2)
        v_flatten = T.flatten(v, 1, 2)

        a_out = self.attention(q_flatten, v_flatten, k_flatten)
    
        # Bring back spacial dimensions: B, q_dim, H, W
        a_out = a_out.view(b, *spatial, c)
        
        # permute back to: B, C, H, W
        a_out = self.permute_layers[1](a_out)

        # gate_images = self.image_forward(values, keys, queries)

        return self.conv(a_out) * self.permute_layers[1](q)
    
        

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, image_shape, pos_encode_kwargs,
                 attn_heads=4, device="cuda", **kwargs):
        super().__init__()
        self.depth=image_shape[-1]
        self.image_shape=image_shape
        self.pos_encode_kwargs=pos_encode_kwargs
        self.attn_heads=attn_heads
        self.permute_layers = kwargs.get("permute_layers", None)
        self.trainable_pe=kwargs.get("trainable_pe", False)
        self.device=device
        
        self.attention = MultiHeadAttention(self.depth, self.depth,
                                            attn_heads=attn_heads,
                                            device=device, **kwargs)

        # positional encoding
        if (self.image_shape is not None):
            if self.trainable_pe:
                self.pos_encode =  T.nn.Parameter(T.randn(*self.image_shape))
            else:
                self.pos_encode = self.positional_encoding(self.image_shape,
                                                    self.pos_encode_kwargs,self.device)
        self.to(self.device)

    def forward(self, q):
        q = self.permute_layers[0](q)
        # Get the shape of the q tensor and save the original for later
        b, *spatial, c = q.shape
        
        # Add positional encoding
        q = q+self.pos_encode.expand_as(q)

        # Flatten each image to combine the spacial dimensions: B, C/F, HxW/point-cloud
        q_flatten = T.flatten(q, 1, 2)

        a_out = self.attention(q_flatten, q_flatten, q_flatten)
    
        # Bring back spacial dimensions: B, q_dim, H, W
        # a_out = a_out.view(b, -1, *spatial)
        a_out = a_out.view(b, *spatial, c)

        # Return the additive connection to the original q
        return self.permute_layers[1](a_out)

    
class VisionTransformerLayer(nn.Module):
    """
    implementation of ViT and T2TViT

    Need to have difference between channel features and patch features
        trainable postional encoding 3d

    First attention in patch then between patches?

    """
    def __init__(self, input_shape:np.ndarray,
                 model_dim:int,
                 trans_enc_cfg: dict,
                 conv_cfg:dict,
                 ctxt_dim:int=0,
                 FiLM_dim:int=0,
                 n_patches:int=None,
                 unfold_cfg:dict={},
                trainable_pe:bool=True,
                device:str="cpu",
                
                **kwargs,
                    ):
        super().__init__()

        self.trans_enc_cfg=trans_enc_cfg
        self.trainable_pe=trainable_pe
        self.n_patches=n_patches
        self.model_dim=model_dim
        self.device=device
        self.ctxt_dim=ctxt_dim
        self.FiLM_dim=FiLM_dim
        self.conv_cfg=conv_cfg

        if (n_patches is None) and (unfold_cfg.get('kernel_size') is None):
            raise ValueError("either n_patches or kernel_size has to be defined")
        
        if isinstance(input_shape, (int, np.int64)):
            self.input_shape = np.array([input_shape,input_shape])
        else:
            self.input_shape=np.array(input_shape)

        self.get_network()
    
    def get_network(self):

        self.kernel_size = np.array(self.input_shape[1:])/(self.n_patches)
        
        if any(self.kernel_size!=np.int64(self.kernel_size)):
            raise ValueError("Image not divisible - change n_patches")
        self.kernel_size = np.int64(self.kernel_size)

        self.patch_features=np.int64(np.product(self.kernel_size))

        self.input_features = self.patch_features*self.input_shape[0]

        if self.trainable_pe:
            # image in the forward will be [B, H, W, C]
           self.pe = T.nn.Parameter(T.randn(*self.input_shape[::-1]))

        # fold/unfolding
        self.args = {"dilation":1, "padding":0, 
                'kernel_size': tuple(self.kernel_size), 
                'stride': self.kernel_size}
    
        self.unfold = nn.Unfold(**self.args)

        self.fold = nn.Fold(output_size=tuple(self.input_shape)[1:], **self.args)
        
        # input embedding
        # self.img_emb= nn.Linear(self.input_features, self.model_dim)
        self.img_emb= nn.Conv2d(in_channels=self.input_shape[0], kernel_size=self.args['kernel_size'],**self.conv_cfg)

        # output embedding
        # self.out_emb = nn.Linear(self.model_dim, self.input_features)
        self.conv_cfg['in_channels']=self.conv_cfg.pop('out_channels')
        self.out_emb= nn.Conv2d(out_channels=self.input_shape[0], kernel_size=self.args['kernel_size'],**self.conv_cfg)

        if self.unfold.stride.sum()*self.conv_cfg['in_channels'] != self.model_dim:
            raise ValueError("Model dimension has to be equal to the unfolded image")
        self.transformer_layer = self.trans_enc_cfg(model_dim=self.model_dim, ctxt_dim=self.ctxt_dim, FiLM_dim=self.FiLM_dim)

        
        self.out_emb.weight.data.fill_(0.00)
        self.out_emb.bias.data.fill_(0.00)

        self.to(self.device)

    def forward(self, image:T.Tensor, ctxt:T.Tensor=None, FiLM:T.Tensor=None, **kwargs) -> T.Tensor:

        image_orig = image.clone()

        # add positional encoding
        if self.trainable_pe:
           image = image+self.pe.expand_as(image)

        # prepare image
        image_processed = self.unfold(self.img_emb(image.permute(0,-1,1,2))).permute(0,-1,1)

        # downscale
        # image_processed = self.img_emb(image_processed)
        
        # dummy mask_vk
        mask_vk = T.ones(image_processed.shape[:-1], dtype=bool, device=image_processed.device)

        # transformer embedding
        attn_image = self.transformer_layer(image_processed, mask_vk=mask_vk, ctxt=ctxt,
                                            FiLM=FiLM)

        # upscale
        # attn_image = self.out_emb(attn_image)

        # fold image back
        output_image = self.out_emb(self.fold(attn_image.permute(0,-1,1))).permute(0,2,3,1)
        
        return output_image+image_orig
    
if __name__ == '__main__':
    n_patches=7
    input_shape = [1, 28,28]
    kernel_size = np.int64(np.array(input_shape[1:])/(n_patches))
    args = {"dilation":1, "padding":0, 'kernel_size': tuple(kernel_size), 'stride': kernel_size}
    
    unfold = nn.Unfold(**args)
    fold = nn.Fold(output_size=tuple(input_shape)[1:], **args)
    
    input_args = T.randn(10, *input_shape)
    
    out = unfold(input_args)

    inpt = fold(out)
                                   
