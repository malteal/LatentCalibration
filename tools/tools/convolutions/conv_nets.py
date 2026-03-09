"UNet"
import math
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from omegaconf import OmegaConf
import torchvision

# internal 
from tools import misc
from tools.discriminator import DenseNet
from tools.modules import Gate, ResidualBlock, FiLM
# from tools.convolutions.image_transformer import (MultiHeadSelfAttention,
#                                                   MultiHeadGateAttention,
#                                                   VisionTransformerLayer)
import tools.positional_encoding as pe


class UNet(nn.Module):
    def __init__(self, input_shape, channels, block_depth, min_size,
                 FiLM_dim=0, ctxt_dim=0, use_gate=True, img_enc=1,
                 dropout=0, device="cuda", **kwargs) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.ctxt_dim = ctxt_dim
        self.FiLM_dim=FiLM_dim
        self.channels=channels
        self.img_enc=img_enc
        self.device=device
        self.block_depth=block_depth
        self.use_gate=use_gate
        self.dropout=dropout
        
        # image dimensions after pooling
        self.img_dims = self.input_shape[-1]//2**(np.arange(len(self.channels)-1))

        if self.img_dims[-1] < min_size:
            raise ValueError("Reduce the number of channels or decrease min_size")

        self.get_network()
        self.to(self.device)

    @T.no_grad()
    def ema(self, state_dict, ema_ratio):
        ema_state_dict = self.state_dict()
        for (key, weight), (em_key, ema_para) in zip(state_dict.items(),
                                                     ema_state_dict.items()):
            ema_state_dict[em_key] = ema_ratio * ema_para + (1 - ema_ratio) * weight

        self.load_state_dict(ema_state_dict)

    def count_trainable_parameters(self):
        sum_trainable = np.sum([i.numel() for i in self.parameters() if i.requires_grad])
        return sum_trainable
        
    def get_network(self):
        self.down_blocks = nn.ModuleList([])
        self.residual_block = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.init_end_conv2 = nn.ModuleList([])
        # self.gates = nn.ModuleList([]) if self.use_gate else None
        self.down_film = nn.ModuleList([])
        self.up_film = nn.ModuleList([])
        
        # permute layer because torch is dumb
        self.in_permute = torchvision.ops.Permute([0,3,1,2])
        self.out_permute = torchvision.ops.Permute([0,2,3,1])

        # Upscale/pooling
        # self.noise_upscale = nn.Upsample(size=self.input_shape[1:], mode="nearest")
        self.upscale_embedding = nn.Upsample(scale_factor=2, mode="nearest")
        self.downscale_embedding = nn.AvgPool2d(2)

        # init conv
        self.start_conv = nn.Conv2d(self.input_shape[0], self.channels[0], kernel_size=1)
        ## Upscale network
        self.end_conv = nn.Conv2d(self.channels[0], self.input_shape[0],kernel_size=1)

        self.end_conv.weight.data.fill_(0.00)
        self.end_conv.bias.data.fill_(0.00)

        ## downscale network
        for nr in range(len(self.channels)-1):
            self.down_blocks.append(
                ResidualBlock(self.channels[nr],self.channels[nr+1],
                              self.block_depth, dropout=self.dropout))
            self.down_film.append(FiLM(self.FiLM_dim, self.channels[nr+1],
                                       device=self.device))

        # upscale part
        for nr in range(len(self.channels)-1):
            self.up_blocks.append(ResidualBlock(2*self.channels[::-1][nr], 
                                                self.channels[::-1][nr+1],
                                                self.block_depth))

            self.up_film.append(FiLM(self.FiLM_dim, self.channels[::-1][nr+1], device=self.device))

    def forward(self, inpt, ctxt=None, FiLM=None, **kwargs):
        if ctxt is None:
            ctxt = {}
        
        # init conv
        x = self.start_conv(self.in_permute(inpt))

        # downscale part
        skips = []
        for down_blk, film in zip(self.down_blocks, self.down_film):
            x = down_blk(x, film(FiLM))
            skips.append(x)
            x = self.downscale_embedding(x)
            
        #upscale part
        for up_blk, film in zip(self.up_blocks, self.up_film):
            skip = skips.pop()

            x = self.upscale_embedding(x)

            x = T.concat([x, skip],1)
            x = up_blk(x, film(FiLM))

        x = self.out_permute(self.end_conv(x))

        return x


if __name__ == "__main__":
    # test
    model = UNet(input_shape=[3,128,128], channels=[16,32,64],
                 block_depth=2, min_size=16, FiLM_dim=32,
                 ctxt_dim=0, use_gate=True, img_enc=1, dropout=0, device="cuda")
    device='cuda'
    print(model)
    print(model.count_trainable_parameters())
    print(model(T.randn(10,128,128,3, device=device), FiLM=T.randn(10,32, device=device)))