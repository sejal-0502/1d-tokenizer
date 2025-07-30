# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, List
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import trunc_normal_
from taming.modules.diffusionmodules.model import nonlinearity, ResnetBlock, AttnBlock, Normalize, Upsample


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0 
    omega = np.arange(embed_dim // 2, dtype=float) 
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega)  

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  
    return emb # 2D array

def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid): 
    assert embed_dim % 2 == 0 

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  

    emb = np.concatenate([emb_h, emb_w], axis=1) 
    return emb # 2D array

def get_2d_sincos_pos_embed(embed_dim, grid_size): 
    """
    grid_size: int or (int, int) of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size 
    grid_h = np.arange(grid_size[0], dtype=np.float32) 
    grid_w = np.arange(grid_size[1], dtype=np.float32) 
    grid = np.meshgrid(grid_w, grid_h) 
    grid = np.stack(grid, axis=0) 

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]]) 
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads 
        project_out = not (heads == 1 and dim_head == dim) 

        self.heads = heads 
        self.scale = dim_head ** -0.5 

        self.attend = nn.Softmax(dim = -1) 
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
            
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x 
            x = ff(x) + x 
        return self.norm(x) 
    
def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class EncoderVIT(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
    mlp_dim: int, depth, heads, model_width, num_latent_tokens, token_size, channels: int = 3, dim_head: int = 64, **ignore_kwargs) -> None:
        super().__init__()
        
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size) 
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height 
        self.patch_width = patch_width
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.model_width = model_width
        self.grid_size = image_height // patch_height
        dim = self.model_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        en_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width) 
        self.patch_dim = channels * patch_height * patch_width 

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size), 
            Rearrange('b c h w -> b (h w) c'), 
        )
        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=True)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.apply(init_weights)
        scale = dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.model_width))
        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.model_width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.model_width))        
        self.conv_out = nn.Conv2d(self.model_width, self.token_size, kernel_size=1, bias=True)

    def resize_pos_embedding(self, new_shape):
        orig_posemb_h = self.image_height // self.patch_height
        orig_posemb_w = self.image_width // self.patch_width
        posemb = self.en_pos_embedding
        if orig_posemb_h == new_shape[0] and orig_posemb_w == new_shape[1]:
            return posemb
        posemb = rearrange(posemb, '1 (h w) d -> 1 d h w', h=orig_posemb_h, w=orig_posemb_w)
        posemb = torch.nn.functional.interpolate(posemb, new_shape, mode='bicubic')
        return rearrange(posemb, '1 d h w -> 1 (h w) d')

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = img.shape[0] 
        ft_h, ft_w = img.shape[-2]//self.patch_height, img.shape[-1]//self.patch_width 
        x = self.to_patch_embedding(img) 
        x = x + self.resize_pos_embedding((ft_h, ft_w)) 

        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        latent_tokens = _expand_token(self.latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.transformer(x) 

        latent_tokens = x[:, 1+self.grid_size**2:] 

        latent_tokens = latent_tokens.reshape(batch_size, self.model_width, self.num_latent_tokens, 1) 

        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)

        return latent_tokens

class DecoderVIT(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 z_channels: int, depth: int, heads: int, mlp_dim: int, dim_head: int=64, **ignore_kwargs) -> None:
        super().__init__()

        dim = z_channels
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.patch_embed = PatchEmbed(image_size, patch_size, dim, dim)
        scale = dim ** -0.5
        self.de_pos_embedding = nn.Parameter(scale*torch.randn(self.num_patches, dim))

        
        self.norm = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

        self.to_pixel = nn.Sequential(
            nn.Linear(dim, 3*patch_size**2),
            Rearrange('b (hh ww) (c sh sw) -> b c (hh sh) (ww sw)', 
                      hh=image_height // patch_height, 
                      ww=image_height // patch_height, 
                      sh=patch_size, sw=patch_size))

        self.init_parameters()


    def init_parameters(self):
        if self.de_pos_embedding is not None:
            nn.init.normal_(self.de_pos_embedding, std=0.02)
        for block in self.transformer.layers:
            for module in block:
                if hasattr(module, 'weight') and module.weight is not None:
                    if len(module.weight.shape) > 1:
                        #nn.init.xavier_uniform_(module.weight)
                        trunc_normal_(module.weight, std=.02)    
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        trunc_normal_(self.MLP[0].weight, std=.02)
        trunc_normal_(self.to_pixel[0].weight, std=.02)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_height
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        
        x = z.permute(0, 2, 3, 1).contiguous() # (B, H, W, D)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x + self.de_pos_embedding
        x = self.norm(x)
        x = self.transformer(x) # output (B, N, D)

        x = self.MLP(x)
        x = self.to_pixel(x) # from SimMIM model

        return x


class DecoderDINOCNN(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, num_layers=1, emb_dim=512, num_heads=12, num_mlp_ratio=4, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.emb_dim = emb_dim

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        self.decoder_dino_conv = nn.Conv2d(block_in, 512, kernel_size=1, stride=1, padding=0)
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent 
            

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim*num_mlp_ratio)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        x = h
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.emb_dim)
        x = self.transformer_encoder(x)
        x = x.reshape(x.shape[0], self.last_z_shape[2], self.last_z_shape[3], self.emb_dim)
        x = x.permute(0, 3, 1, 2)

        return h, x