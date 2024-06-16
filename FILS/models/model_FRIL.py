from functools import partial
import math
from einops import pack, rearrange, repeat, unpack
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint

import flash_attn
from flash_attn.modules.mha import MHA as FlashMHA

from FILS.models.transformer import TextTransformer
from FILS.models.utils import enable_grad_checkpointing, remap_keys_from_open_clip_to_vit
from FILS.utils.misc import interpolate_pos_embed
from FILS.models.model_clip import CLIP
import clip

torch_version = torch.__version__
is_torch2 = torch_version.startswith('2.') 


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class WeightedFeatureMaps(nn.Module):
    def __init__(self, k, embed_dim, *, norm_layer=nn.LayerNorm, decoder_depth):
        super(WeightedFeatureMaps, self).__init__()
        self.linear = nn.Linear(k, decoder_depth, bias=False)
        
        std_dev = 1. / math.sqrt(k)
        nn.init.normal_(self.linear.weight, mean=0., std=std_dev)

    def forward(self, feature_maps):
        # Ensure the input is a list
        assert isinstance(feature_maps, list), "Input should be a list of feature maps"
        # Ensure the list has the same length as the number of weights
        assert len(feature_maps) == (self.linear.weight.shape[1]), "Number of feature maps and weights should match"
        stacked_feature_maps = torch.stack(feature_maps, dim=-1)  # shape: (B, L, C, k)
        # compute a weighted average of the feature maps
        # decoder_depth is denoted as j
        output = self.linear(stacked_feature_maps)
        return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # comment this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        if is_torch2:
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """
        query from decoder (x), key and value from encoder (y)
        """
        B, N, C = x.shape
        Ny = y.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if is_torch2:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop,
            )
            x = attn.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, self_attn=False, use_flash_attn=False,):
        super().__init__()
        self.self_attn = self_attn
        if self.self_attn:
            self.norm0 = norm_layer(decoder_dim)
            self.self_attn = Attention(
                decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(decoder_dim)
        if not use_flash_attn:
            self.cross_attn = CrossAttention(
                encoder_dim, decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            # if flash_attn version is less than 1.0.0
            if flash_attn.__version__[0] == '0':
                self.cross_attn = FlashMHA(decoder_dim, num_heads, cross_attn=True, bias=qkv_bias, dropout=attn_drop, use_flash_attn=True)
            else:
                self.cross_attn = FlashMHA(decoder_dim, num_heads, cross_attn=True, qkv_proj_bias=qkv_bias, dropout=attn_drop, use_flash_attn=True)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(decoder_dim)
        mlp_hidden_dim = int(decoder_dim * mlp_ratio)
        self.mlp = Mlp(in_features=decoder_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        """
        x: decoder feature; y: encoder feature (after layernorm)
        """
        if self.self_attn:
            x = x + self.drop_path(self.self_attn(self.norm0(x)))
        x = x + self.drop_path(self.cross_attn(self.norm1(x), y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_flash_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not use_flash_attn:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        else:
            # if flash_attn version is less than 1.0.0
            if flash_attn.__version__[0] == '0':
                self.attn = FlashMHA(dim, num_heads, cross_attn=False, bias=qkv_bias, dropout=attn_drop, use_flash_attn=True)
            else:
                self.attn = FlashMHA(dim, num_heads, cross_attn=False, qkv_proj_bias=qkv_bias, dropout=attn_drop, use_flash_attn=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2, channel_last=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.channel_last = channel_last
        if channel_last:
            self.proj = nn.Linear(in_features=in_chans * tubelet_size * patch_size[0] * patch_size[1], out_features=embed_dim)
        else:
            self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                                kernel_size = (self.tubelet_size,  patch_size[0], patch_size[1]),
                                stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        if self.channel_last:
            x = rearrange(x, 'b c (t p0) (h p1) (w p2) -> b (t h w) (c p0 p1 p2)',
                          p0=self.tubelet_size, p1=self.patch_size[0], p2=self.patch_size[1])
            # x = rearrange(x, 'b (t h w) (p0 p1 p2) c -> b (t h w) (c p0 p1 p2)')
            x = self.proj(x)
            return x
        else:
            B, C, T, H, W = x.shape
            # FIXME look at relaxing size constraints
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 fc_drop_rate=0., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.001,
                 all_frames=16,
                 tubelet_size=2,
                 channel_last=False,
                 use_checkpoint=False,
                 use_flash_attn=False,
                 use_mean_pooling=True,
                 args=None,):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size,
            channel_last=channel_last,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_flash_attn=use_flash_attn)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

        # if registers are enabled
        self.use_registers = args.use_registers
        if args.use_registers:
            self.register_tokens = nn.Parameter(
                torch.randn(args.num_registers, embed_dim)
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.use_registers:
            #repeat register token
            r = repeat(
                self.register_tokens, 
                'n d -> b n d', 
                b=B
            )
            #pack cls token and register token
            x, ps = pack([x, r], 'b * d ')

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:   
            for blk in self.blocks:
                x = blk(x)

        if self.use_registers:
            #unpack cls token and register token
            x, _ = unpack(x, ps, 'b * d')

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x
    
class FRILSVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 fc_drop_rate=0., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.001,
                 all_frames=16,
                 tubelet_size=2,
                 channel_last=False,
                 use_checkpoint=False,
                 use_flash_attn=False,
                 use_mean_pooling=True,
                 text_embed_dim=512,
                 args=None,):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size,
            channel_last=channel_last,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.use_mean_pooling = use_mean_pooling

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_flash_attn=use_flash_attn)
            for i in range(depth)])
        
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.v2t_mapping = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), 
                            act_layer=nn.GELU, drop=0,
                            out_features=int(text_embed_dim))

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # if head is not nn.Identity
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # if registers are enabled
        self.use_registers = args.use_registers if args is not None else False
        if self.use_registers:
            self.register_tokens = nn.Parameter(
                torch.randn(args.num_registers, embed_dim)
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.use_registers:
            #repeat register token
            r = repeat(
                self.register_tokens, 
                'n d -> b n d', 
                b=B
            )
            #pack cls token and register token
            x, ps = pack([x, r], 'b * d ')

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:   
            for blk in self.blocks:
                x = blk(x)

        if self.use_registers:
            #unpack cls token and register token
            x, _ = unpack(x, ps, 'b * d')

        x = self.norm(x)
        x = self.v2t_mapping(x)
        if self.use_mean_pooling:
            return x.mean(1)
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x

class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_flash_attn=False,
                 channel_last=False,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size,
            channel_last=channel_last,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint


        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_flash_attn=use_flash_attn)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis, use_reentrant=False)
        else:   
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class FRILS_PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_flash_attn=False,
                 channel_last=False,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size,
            channel_last=channel_last,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint


        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_flash_attn=use_flash_attn)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        embedded_patch = self.patch_embed(x)
        
        x = embedded_patch + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis, use_reentrant=False)
        else:   
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis, embedded_patch

    def forward(self, x, mask):
        x, embedded_patch = self.forward_features(x, mask)
        x = self.head(x)
        return x, embedded_patch

class FRILS_CrossPretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_flash_attn=False, use_input=False, weight_fm=False, use_fm=[-1],
                 channel_last=False,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.use_input = use_input
        self.weight_fm = weight_fm
        self.use_fm = use_fm
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size,
            channel_last=channel_last,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint


        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_flash_attn=use_flash_attn)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, ids_restore, ids_keep ):
        _, _, T, _, _ = x.shape
        embedded_patch = self.patch_embed(x)
        
        x = embedded_patch + self.pos_embed.type_as(x).to(x.device).clone().detach()

        ####################################
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, embedded_patch.shape[-1]))
        # x = torch.gather(x, dim=1, index=ids_keep.to(x.device).unsqueeze(-1).repeat(1, 1, embedded_patch.shape[-1]))


        # apply Transformer blocks
        x_feats = []
        if self.use_input:
            x_feats.append(x)
        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if self.weight_fm and idx in self.use_fm:
                x_feats.append(x)

        if self.weight_fm:
            return x_feats, embedded_patch
        else:
            x = self.norm(x)
            return x, embedded_patch
        ####################################

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis, use_reentrant=False)
        else:   
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis, embedded_patch

    def forward(self, x, mask, ids_restore, ids_keep):
        x, embedded_patch = self.forward_features(x, mask, ids_restore, ids_keep)
        x = self.head(x)
        return x, embedded_patch


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False,
                 use_flash_attn=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_flash_attn=use_flash_attn)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:   
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class FRILS_PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False,
                 use_flash_attn=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_flash_attn=use_flash_attn)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            pred_features = checkpoint.checkpoint(nn.Identity(), x, use_reentrant=False)
        else:   
            for blk in self.blocks:
                x = blk(x)
            pred_features = x

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x, pred_features

class FRILS_CrossPretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False,
                 use_flash_attn=False, weight_fm=False, use_input=False, use_fm=[-1], self_attn=False, in_chans=3,
                 encoder_embed_dim=768,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        ##################################
        # weighted feature maps for cross attention
        self.weight_fm = weight_fm
        self.use_input = use_input # use input as one of the feature maps
        if len(use_fm) == 1 and use_fm[0] == -1:
            self.use_fm = list(range(depth))
        else:
            self.use_fm = [i if i >= 0 else depth + i for i in use_fm]
        if self.weight_fm:
            # print("Weighting feature maps!")
            # print("using feature maps: ", self.use_fm)
            dec_norms = []
            for i in range(depth):
                norm_layer_i = norm_layer(embed_dim)
                dec_norms.append(norm_layer_i)
            self.dec_norms = nn.ModuleList(dec_norms)

            # feature weighting
            self.wfm = WeightedFeatureMaps(len(self.use_fm) + (1 if self.use_input else 0), embed_dim, norm_layer=norm_layer, decoder_depth=depth)

        ##################################
        print("use self attention: ", self_attn)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(encoder_embed_dim, embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, self_attn=self_attn, use_flash_attn=use_flash_attn)
            for i in range(depth)])

        ##################################

        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes, bias=True) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, y, return_token_num, mask, ids_restore,):
        ###################################
        # N, L = ids_restore.shape

        # # contruct mask tokens 
        # y = self.decoder_pos_embed[:, :].masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1, self.mask_token.shape[-1])
        # y = y + self.mask_token

        if self.weight_fm:
            # x input: a list of Tensors (B, C, D)
            x = self.wfm(x)

        for i, blk in enumerate(self.blocks):
            if self.weight_fm:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, y, self.dec_norms[i](y[..., i]), use_reentrant=False)
                else:
                    x = blk(y, self.dec_norms[i](y[..., i]))
                pred_features = checkpoint.checkpoint(nn.Identity(), x, use_reentrant=False)
            else:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, y, x, use_reentrant=False)
                else:
                    x = blk(y, x)
                pred_features = x
        ###################################

        # if self.use_checkpoint:
        #     for blk in self.blocks:
        #         x = checkpoint.checkpoint(blk, x, use_reentrant=False)
        #     pred_features = checkpoint.checkpoint(nn.Identity(), x, use_reentrant=False)
        # else:   
        #     for blk in self.blocks:
        #         x = blk(x)
        #     pred_features = x

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x, pred_features


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 use_flash_attn_at_encoder=False,
                 use_flash_attn_at_decoder=False,
                 tubelet_size=2,
                 channel_last=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn_at_encoder,
            use_learnable_pos_emb=use_learnable_pos_emb,
            channel_last=channel_last,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn_at_decoder)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return x

class FRILS_PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 use_flash_attn_at_encoder=False,
                 use_flash_attn_at_decoder=False,
                 tubelet_size=2,
                 channel_last=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 text_embed_dim=512,#768 #512(vifi) #1280(OpenCLIP)
                  ):
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.text_embed_dim = text_embed_dim
        self.encoder = FRILS_PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn_at_encoder,
            use_learnable_pos_emb=use_learnable_pos_emb,
            channel_last=channel_last,
        )

        self.decoder = FRILS_PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn_at_decoder)
        
        self.v2t_mapping = Mlp(in_features=encoder_embed_dim, hidden_features=int(encoder_embed_dim * mlp_ratio), 
                            act_layer=nn.GELU, drop=0,
                            out_features=int(text_embed_dim))
        
        self.t_mapping = Mlp(in_features=decoder_embed_dim, hidden_features=int(decoder_embed_dim * mlp_ratio), 
                            act_layer=nn.GELU, drop=0,
                            out_features=int(encoder_embed_dim))

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        # clip parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}
    
    def encode(self, x):
        tensors = torch.zeros((x.shape[0], 196))
        indexes = torch.argsort(torch.rand(x.shape[0], 196))[:, :176]
        tensors[torch.arange(x.shape[0]).unsqueeze(1), indexes] = 1
        mask = tensors.repeat(1, 8)
        mask.cuda(x.device, non_blocking=True)
        mask = mask.flatten(1).to(torch.bool)

        _, embedded_patch = self.encoder(x, mask)
        mapped_embedded_patch = self.v2t_mapping(embedded_patch)

        # return mapped_embedded_patch.mean(1)
        return mapped_embedded_patch

    def forward(self, x, mask):
        b, _, T, _, _ = x.shape
        x_vis, embedded_patch = self.encoder(x, mask) # [B, N_vis, C_e]
        mapped_embedded_patch = self.v2t_mapping(embedded_patch) # [B, N_vis, C_e] # map the visual feature to the text feature space
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x, pred_feature = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        # make the pred_feature dim equal to the text feature dim
        pred_feature = self.t_mapping(pred_feature)

        mapped_masked_embedded_patch = mapped_embedded_patch[mask].reshape(b, -1, self.text_embed_dim)
        mapped_masked_pred_feature = self.v2t_mapping(pred_feature[mask].reshape(b, -1, self.encoder_embed_dim))

        return x, embedded_patch, mapped_embedded_patch, pred_feature, mapped_masked_embedded_patch, mapped_masked_pred_feature, self.logit_scale.exp()

class FRILS_CrossPretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 use_flash_attn_at_encoder=False,
                 use_flash_attn_at_decoder=False,
                 tubelet_size=2,
                 channel_last=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 text_embed_dim=768,
                 weight_fm=False, 
                 use_input=False, 
                 use_fm=[-1]
                 ):
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.text_embed_dim = text_embed_dim
        self.encoder = FRILS_CrossPretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn_at_encoder,
            use_learnable_pos_emb=use_learnable_pos_emb,
            channel_last=channel_last,
            weight_fm=weight_fm, 
            use_input=use_input, 
            use_fm=use_fm,
        )

        self.decoder = FRILS_CrossPretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn_at_decoder,
            in_chans=encoder_in_chans,
            weight_fm=weight_fm, 
            use_input=use_input, 
            use_fm=use_fm,
            encoder_embed_dim=encoder_embed_dim,
            )
        
        self.v2t_mapping = Mlp(in_features=encoder_embed_dim, hidden_features=int(encoder_embed_dim * mlp_ratio), 
                            act_layer=nn.GELU, drop=0,
                            out_features=int(encoder_embed_dim))
        
        self.t_mapping = Mlp(in_features=decoder_embed_dim, hidden_features=int(decoder_embed_dim * mlp_ratio), 
                            act_layer=nn.GELU, drop=0,
                            out_features=int(text_embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
        
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        trunc_normal_(self.mask_token, std=.02)

        # clip parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, ids_restore, ids_keep):
        b, _, T, _, _ = x.shape
        x_vis, embedded_patch = self.encoder(x, mask, ids_restore, ids_keep) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]

        mapped_embedded_patch = self.v2t_mapping(embedded_patch) # [B, N_vis, C_e] # map the visual feature to the text feature space
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        # x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        # x, pred_feature = self.decoder(x_full, pos_emd_mask.shape[1], mask, ids_restore) # [B, N_mask, 3 * 16 * 16]
        x, pred_feature = self.decoder(x=x_vis, y=(self.mask_token + pos_emd_mask), return_token_num=pos_emd_mask.shape[1], mask=mask, ids_restore=ids_restore) # [B, N_mask, 3 * 16 * 16]

        # make the pred_feature dim equal to the text feature dim
        pred_feature = self.t_mapping(pred_feature)

        mapped_masked_embedded_patch = mapped_embedded_patch[mask].reshape(b, -1, self.text_embed_dim)
        mapped_masked_pred_feature = self.v2t_mapping(pred_feature.reshape(b, -1, self.text_embed_dim))

        return x, embedded_patch, mapped_embedded_patch, pred_feature, mapped_masked_embedded_patch, mapped_masked_pred_feature, self.logit_scale.exp()


def VIDEOMAE_VITB16(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


def FRILS_VITB16(pretrained=False, **kwargs):
    model = FRILS_PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384, # 768
        decoder_num_heads=6, # 12
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

def FRILSCross_VITB32(pretrained=False, **kwargs):
    model = FRILS_CrossPretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def load_frils_visual_model(model, pretrain_path, use_flash_attn):
    if pretrain_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(pretrain_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
    print("=> Load checkpoint from %s" % pretrain_path)
    for model_key in "model|module|state_dict".split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            if list(checkpoint_model.keys())[0].startswith('module.'):
                renamed_ckpt = {k[7:]: v for k, v in checkpoint_model.items()}
                checkpoint_model = renamed_ckpt
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    for key in ['head.weight', 'head.bias']: ## modify here to remove extra keys
        if key in checkpoint_model and checkpoint_model[key].shape != model.state_dict()[key].shape:
            print("Removing key %s from pretrained checkpoint" % key)
            checkpoint_model.pop(key)

    # new_dict = OrderedDict()
    # for key in checkpoint_model.keys():
    #     if key.startswith('backbone.'):
    #         new_dict[key[9:]] = checkpoint_model[key]
    #     elif key.startswith('encoder.'):
    #         if use_flash_attn and 'attn.qkv' in key:
    #             new_dict[key[8:].replace('attn.qkv', 'attn.Wqkv')] = checkpoint_model[key]
    #         elif use_flash_attn and 'attn.q_bias' in key:
    #             q_bias = checkpoint_model[key]
    #             v_bias = checkpoint_model[key.replace('attn.q_bias', 'attn.v_bias')]
    #             new_dict[key[8:].replace('attn.q_bias', 'attn.Wqkv.bias')] = torch.cat(
    #                 (q_bias, torch.zeros_like(v_bias), v_bias))
    #         elif use_flash_attn and 'attn.v_bias' in key:
    #             continue
    #         elif use_flash_attn and 'attn.proj' in key:
    #             new_dict[key[8:].replace('attn.proj', 'attn.out_proj')] = checkpoint_model[key]
    #         else:
    #             new_dict[key[8:]] = checkpoint_model[key]
    #     else:
    #         if use_flash_attn and 'attn.qkv' in key:
    #             new_dict[key.replace('attn.qkv', 'attn.Wqkv')] = checkpoint_model[key]
    #         elif use_flash_attn and 'attn.q_bias' in key:
    #             q_bias = checkpoint_model[key]
    #             v_bias = checkpoint_model[key.replace('attn.q_bias', 'attn.v_bias')]
    #             new_dict[key.replace('attn.q_bias', 'attn.Wqkv.bias')] = torch.cat(
    #                 (q_bias, torch.zeros_like(v_bias), v_bias))
    #         elif use_flash_attn and 'attn.v_bias' in key:
    #             continue
    #         elif use_flash_attn and 'attn.proj' in key:
    #             new_dict[key.replace('attn.proj', 'attn.out_proj')] = checkpoint_model[key]
    #         else:
    #             new_dict[key] = checkpoint_model[key]
    # checkpoint_model = new_dict

    if 'pos_embed' in checkpoint_model:
        new_pos_embed = interpolate_pos_embed(checkpoint_model['pos_embed'], model, num_frames=16)
        checkpoint_model['pos_embed'] = new_pos_embed

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)
    logit_scale = checkpoint_model['logit_scale']
    
    # delet unneccessary parameters
    del checkpoint, checkpoint_model
    
    # return the logit_scale
    return logit_scale

def FRILSCLIP_VITB16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    text_pretrain_path=None,
    **kwargs
):
    # vision_model = FRILSVisionTransformer(
    #     patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    #     num_classes=0,
    #     fc_drop_rate = 0,
    #     drop_rate = 0,
    #     drop_path_rate=0.1,
    #     attn_drop_rate=0,
    #     use_flash_attn=use_flash_attn,
    #     use_checkpoint=use_grad_checkpointing,
    #     channel_last=False,
    #     text_embed_dim=project_embed_dim,
    # )
    # initiate the visual model
    if pretrain_zoo == "frils":
        vision_model = FRILS_VITB16(
            pretrained=False,
            drop_path_rate=0.1,
            decoder_depth=6,
            use_flash_attn_at_encoder=use_flash_attn,
            use_flash_attn_at_decoder=use_flash_attn,
            use_checkpoint=use_grad_checkpointing,
            channel_last=False,
            text_embed_dim=project_embed_dim,
        )
    else:
        vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
        enable_grad_checkpointing(vision_model, use_grad_checkpointing)

    # initiate the text model
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=project_embed_dim, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    
    # initiate the CLIP model
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(state_dict, use_fast_conv1=use_fast_conv1, use_flash_attn=use_flash_attn)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "frils":
        assert pretrain_path is not None
        print("=> loading frils model")
        # update the visual state_dict
        logit_scale = load_frils_visual_model(model.visual, pretrain_path, use_flash_attn)
        # update the logit_scale in CLIP
        model.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        
        # pre-initialize the text model with the openai model
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        # clip_model, _ = clip.load("ViT-L/14", device='cpu')
        missing_keys, unexpected_keys = model.textual.load_state_dict(clip_model.state_dict(), strict=False)
        del clip_model
        
        # update the text state_dict with the fine-tuned one
        text_state_dict = torch.load(text_pretrain_path)['model']
        # now remove the unwanted keys:
        if "module.prompt_learner.token_prefix" in text_state_dict:
            del text_state_dict["module.prompt_learner.token_prefix"]

        if "module.prompt_learner.token_suffix" in text_state_dict:
            del text_state_dict["module.prompt_learner.token_suffix"]

        if "module.prompt_learner.complete_text_embeddings" in text_state_dict:
            del text_state_dict["module.prompt_learner.complete_text_embeddings"]
        # remove the prefix "module."
        text_state_dict = {k.replace("module.", ""): v for k, v in text_state_dict.items()}
        # remove the prefix "text_encoder."
        text_state_dict = {k.replace("text_encoder.", ""): v for k, v in text_state_dict.items()}
        missing_keys, unexpected_keys = model.textual.load_state_dict(text_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model
