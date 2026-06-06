import math
from functools import partial

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc

try:
    from utils.dcn import DeformableConv2d
except ImportError as exc:
    DeformableConv2d = None
    _DCN_IMPORT_ERROR = exc


def require_class_names(class_names):
    if class_names is None or len(class_names) == 0:
        raise ValueError("class_names must be provided, e.g. class_names=RS.ST_CLASSES.")
    return list(class_names)

# =========================================================================
#  Backbone & Basic Blocks 
# =========================================================================

class Backbone(nn.Module):
    def __init__(self, patch_size=7, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes    = num_classes
        self.depths         = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]     
        
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        cur = 0
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])
        
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])
        
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        x, H, W = self.patch_embed1.forward(x)
        for i, blk in enumerate(self.block1):
            x = blk.forward(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        x, H, W = self.patch_embed2.forward(x)
        for i, blk in enumerate(self.block2):
            x = blk.forward(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        x, H, W = self.patch_embed3.forward(x)
        for i, blk in enumerate(self.block3):
            x = blk.forward(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        x, H, W = self.patch_embed4.forward(x)
        for i, blk in enumerate(self.block4):
            x = blk.forward(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self,patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=False):
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim        = dim
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5
        self.q          = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr     = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm   = nn.LayerNorm(dim)
        self.kv         = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# =========================================================================
#  CLIP & CoOp Modules
# =========================================================================

class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, n_ctx=16, ctx_init="satellite view of"):
        super().__init__()
        n_cls = len(class_names)
        dtype = torch.float32 
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = clip_model.token_embedding.weight.device

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        class_names = [name.replace("_", " ") for name in class_names]
        prompts = [prompt_prefix + " " + name + "." for name in class_names]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.register_buffer('tokenized_prompts', tokenized_prompts)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

class CLIPGuidedTextFeatureInjector(nn.Module):
    def __init__(self, class_names, feature_dim=256, 
                 model_path='/home/zhj/TSFDNet/pretrained/RemoteCLIP-ViT-B-32.pt'):
        super().__init__()
        class_names = require_class_names(class_names)
        self.class_names = class_names
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.attn_temp = nn.Parameter(torch.tensor(1.0))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model_name = "ViT-B/32"
        
        print(f"Loading CLIP {clip_model_name} from {model_path}...")
        
        try:
            model, _ = clip.load(clip_model_name, device='cpu', jit=False)
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"CLIP Custom Weights Loaded: {msg}")
            self.clip_model = model.to(device)
        except Exception as e:
            print(f"Error loading custom weights: {e}. Falling back to default OpenAI weights.")
            self.clip_model, _ = clip.load(clip_model_name, device=device, jit=False)

        self.clip_model.float() 
        
        # 冻结 CLIP 参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.transformer.resblocks[-1].parameters():
            param.requires_grad = True
        for param in self.clip_model.ln_final.parameters():
            param.requires_grad = True

        self.prompt_learner = PromptLearner(class_names, self.clip_model)
        self.text_dim = self.clip_model.ln_final.weight.shape[0]
        self.visual_adapter = nn.Conv2d(feature_dim, self.text_dim, kernel_size=1)
        self.channel_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(len(class_names), feature_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gamma_spatial = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.gamma_channel = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.project_back = nn.Sequential(
            nn.Conv2d(self.text_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.float() 
        prompts = self.prompt_learner()
        
        text_features = prompts + self.clip_model.positional_embedding.float()
        text_features = text_features.permute(1, 0, 2)
        text_features = self.clip_model.transformer(text_features)
        text_features = text_features.permute(1, 0, 2)
        text_features = self.clip_model.ln_final(text_features).float()

        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = text_features[torch.arange(text_features.shape[0]), tokenized_prompts.argmax(dim=-1)]
        text_features = text_features @ self.clip_model.text_projection.float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        x_clip_space = F.normalize(self.visual_adapter(x), dim=1)
        text_features = F.normalize(text_features, dim=-1)
        scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        temp  = self.attn_temp.clamp(0.1, 10.0)
        sim_map = torch.einsum('kd, bdhw -> bkhw', text_features, x_clip_space) * scale
        spatial_attn = F.softmax(sim_map / temp, dim=1)

        text_context = torch.einsum('bkhw, kd -> bdhw', spatial_attn, text_features)
        spatial_out = self.project_back(text_context)
        
        channel_scale = self.channel_adapter(sim_map)
        
        out = x + \
              self.gamma_spatial * spatial_out + \
              self.gamma_channel * (x * channel_scale)
        return out

class SemanticGuidedSpatialEnhancement(nn.Module):
    def __init__(self):
        super(SemanticGuidedSpatialEnhancement, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_normal_(self.conv2d.weight)
        if self.conv2d.bias is not None:
            nn.init.constant_(self.conv2d.bias, 0)
    def forward(self, x):
        x_s = x[0]
        x_c = x[1]
        avgout = torch.mean(x_s, dim=1, keepdim=True)
        maxout, _ = torch.max(x_s, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = x_c * out
        return out

class FeatureProjection(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(p=drop)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = self.dropout(x)
        return x

# =========================================================================
#  Wavelet Frequency Decomposition
# =========================================================================

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required by MambaLayer.") from _MAMBA_IMPORT_ERROR
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, 
                d_state=d_state,  
                d_conv=d_conv,    
                expand=expand,    
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.mamba(x)
        return x

class CrossScanLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def scan(self, x):
        B, C, H, W = x.shape
        xs = [
            x,
            torch.flip(x, dims=[-1, -2]),
            torch.flip(x, dims=[-1]),
            torch.flip(x, dims=[-2]),
        ]
        xs = [t.flatten(2).transpose(1, 2) for t in xs]
        x_scan = torch.cat(xs, dim=0)
        return x_scan

    def merge(self, x_scan, H, W):
        B_4, L, C = x_scan.shape
        B = B_4 // 4
        v1, v2, v3, v4 = torch.split(x_scan, B, dim=0)
        v1 = v1.transpose(1, 2).view(B, C, H, W)
        v2 = v2.transpose(1, 2).view(B, C, H, W)
        v3 = v3.transpose(1, 2).view(B, C, H, W)
        v4 = v4.transpose(1, 2).view(B, C, H, W)
        v2 = torch.flip(v2, dims=[-1, -2])
        v3 = torch.flip(v3, dims=[-1])
        v4 = torch.flip(v4, dims=[-2])
        return (v1 + v2 + v3 + v4) / 4.0

class HaarDWT(nn.Module):
    def __init__(self, in_channels):
        super(HaarDWT, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                          hl[None,::-1,::-1], hh[None,::-1,::-1]], axis=0)
        filts = np.tile(filts, (in_channels, 1, 1, 1))
        weight = torch.tensor(filts, dtype=torch.float32)
        self.register_buffer('weight', weight)

    def dwt(self, x):
        return F.conv2d(x, self.weight, padding=0, stride=2, groups=x.shape[1])

class WaveletFrequencyDecomposition(nn.Module):
    def __init__(self, in_channels, out_channels=None, H=32, W=32):
        super().__init__()
        if DeformableConv2d is None:
            raise ImportError("utils.dcn.DeformableConv2d is required by WaveletFrequencyDecomposition.") from _DCN_IMPORT_ERROR
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        
        self.dwt_layer = HaarDWT(in_channels)
        self.scanner = CrossScanLayer()
        self.mamba_fusion = MambaLayer(dim=in_channels * 2) 
        
        self.high_freq_compress = nn.Sequential(
            nn.Conv2d(in_channels * 6, in_channels * 2, kernel_size=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.high_freq_norm = nn.BatchNorm2d(in_channels * 2)
        self.high_freq_alpha = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.high_freq_dcn = DeformableConv2d(
            in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=3, padding=1
        )
        self.high_freq_act = nn.Sequential(
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )

        self.z_proj = nn.Linear(in_channels * 2, in_channels * 2) 
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )
        
        self.diff_embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels), 
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1) 
        )
        
        self.diff_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.diff_residual = nn.Sequential(
             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
             nn.BatchNorm2d(in_channels),
             nn.ReLU(inplace=True)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_list):
        t1, t2 = x_list[0], x_list[1]
        B, C, H, W = t1.shape

        diff_raw = torch.abs(t1 - t2)
        diff_feat = self.diff_embed(diff_raw) 
        
        gate_mask = self.diff_gate(diff_feat)
        diff_feat_gated = diff_feat * gate_mask 
        
        t1_enhanced = t1 + diff_feat_gated
        t2_enhanced = t2 + diff_feat_gated

        t1_dwt = self.dwt_layer.dwt(t1_enhanced) 
        t2_dwt = self.dwt_layer.dwt(t2_enhanced)
        
        t1_LL = t1_dwt[:, :C, :, :]
        t2_LL = t2_dwt[:, :C, :, :]
        t1_High = t1_dwt[:, C:, :, :]
        t2_High = t2_dwt[:, C:, :, :]

        ll_concat = torch.cat([t1_LL, t2_LL], dim=1)
        H_ll, W_ll = ll_concat.shape[2], ll_concat.shape[3]
        
        x_scanned = self.scanner.scan(ll_concat)
        global_feat_scanned = self.mamba_fusion(x_scanned)
        global_feat_2d = self.scanner.merge(global_feat_scanned, H_ll, W_ll)

        high_concat = torch.cat([t1_High, t2_High], dim=1)
        local_feat = self.high_freq_compress(high_concat)
        local_feat = self.high_freq_norm(local_feat)
        local_feat = self.high_freq_dcn(local_feat)
        local_feat = self.high_freq_act(local_feat)

        gate = torch.sigmoid(self.z_proj(global_feat_2d.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        
        fused_feat_small = global_feat_2d * gate + local_feat * self.high_freq_alpha 
        fused_feat = F.interpolate(fused_feat_small, size=(H, W), mode='bilinear', align_corners=False)
        out_fused = self.out_proj(fused_feat)

        diff_res = self.diff_residual(diff_raw)
        
        return out_fused + diff_res

# =========================================================================
#  Prediction Heads & Decoder
# =========================================================================

def make_prediction(in_channels, out_channels, sigmoid=False):
    if sigmoid:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SemanticHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SemanticHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = FeatureProjection(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = FeatureProjection(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = FeatureProjection(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = FeatureProjection(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x,_c

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class ChangeGuidedSemanticAttention(nn.Module):
    def __init__(self, semantic_dim, change_feat_dim):
        super(ChangeGuidedSemanticAttention, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(semantic_dim + change_feat_dim, semantic_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(semantic_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(semantic_dim, semantic_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, semantic_logits, change_feats):
        if semantic_logits.shape[2:] != change_feats.shape[2:]:
            change_feats = F.interpolate(change_feats, size=semantic_logits.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([semantic_logits, change_feats], dim=1)
        attn_mask = self.fusion(fused)
        return semantic_logits + semantic_logits * attn_mask

class TSFDDecoder(nn.Module):
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                 in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=9, 
                 decoder_softmax = False, feature_strides=[2, 4, 8, 16],
                 class_names=None,
                 img_size=512,
                 clip_head_path='/home/zhj/TSFDNet/pretrained/RemoteCLIP-ViT-B-32.pt'): 
        super(TSFDDecoder, self).__init__()
        class_names = require_class_names(class_names)
        if output_nc != len(class_names):
            raise ValueError(f"output_nc ({output_nc}) must match len(class_names) ({len(class_names)}).")
        
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = FeatureProjection(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = FeatureProjection(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = FeatureProjection(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = FeatureProjection(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        self.text_feature_injector = CLIPGuidedTextFeatureInjector(
            class_names, feature_dim=self.embedding_dim, model_path=clip_head_path
        )
        
        H_in, W_in = img_size, img_size
        self.wfd_semantic = WaveletFrequencyDecomposition(in_channels=self.embedding_dim, H=H_in//4, W=W_in//4)
        self.wfd_c4 = WaveletFrequencyDecomposition(in_channels=self.embedding_dim, H=H_in//32, W=W_in//32)
        self.wfd_c3 = WaveletFrequencyDecomposition(in_channels=self.embedding_dim, H=H_in//16, W=W_in//16)
        self.wfd_c2 = WaveletFrequencyDecomposition(in_channels=self.embedding_dim, H=H_in//8, W=W_in//8)
        self.wfd_c1 = WaveletFrequencyDecomposition(in_channels=self.embedding_dim, H=H_in//4, W=W_in//4)

        self.sgse = SemanticGuidedSpatialEnhancement()
        self.in_channels = [64, 128, 320, 512]
        self.decode_head = SemanticHead(self.output_nc, self.in_channels, self.embedding_dim)
        
        self.make_pred_c4_bcd = make_prediction(in_channels=self.embedding_dim, out_channels=1,sigmoid = True)
        self.make_pred_c3_bcd = make_prediction(in_channels=self.embedding_dim, out_channels=1,sigmoid = True)
        self.make_pred_c2_bcd = make_prediction(in_channels=self.embedding_dim, out_channels=1,sigmoid = True)
        self.make_pred_c1_bcd = make_prediction(in_channels=self.embedding_dim, out_channels=1,sigmoid = True)
        self.make_pred_s_bcd = make_prediction(in_channels=self.embedding_dim, out_channels=1,sigmoid = True)
        self.make_pred_bcd = make_prediction(in_channels=self.embedding_dim, out_channels=1,sigmoid = True)

        self.linear_fuse_bcd = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * 5, out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))

        self.cgsa_p1 = ChangeGuidedSemanticAttention(semantic_dim=output_nc, change_feat_dim=embedding_dim)
        self.cgsa_p2 = ChangeGuidedSemanticAttention(semantic_dim=output_nc, change_feat_dim=embedding_dim)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for m in self.modules():
            if isinstance(m, (CLIPGuidedTextFeatureInjector, FeatureProjection, WaveletFrequencyDecomposition)):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs1, inputs2):
        x_1 = self._transform_inputs(inputs1)
        x_2 = self._transform_inputs(inputs2)

        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        p_1, s_c1 = self.decode_head([c1_1, c2_1, c3_1, c4_1])
        p_2, s_c2 = self.decode_head([c1_2, c2_2, c3_2, c4_2])
        semantic_diff = torch.abs(s_c1 - s_c2)
        
        n, _, h, w = c4_1.shape
        outputs = []
        
        diff_s = self.wfd_semantic([s_c1, s_c2])
        p_bcd_s  = self.make_pred_s_bcd(diff_s)
        
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4_1 = self.text_feature_injector(_c4_1)
        _c4_2 = self.text_feature_injector(_c4_2)
        diff_feat_c4 = self.wfd_c4([_c4_1, _c4_2])
        diff_feat_c4 = self.sgse([resize(diff_s, size=diff_feat_c4.size()[2:], mode='bilinear', align_corners=False),diff_feat_c4 ])
        p_bcd_c4  = self.make_pred_c4_bcd(diff_feat_c4)
        
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3_1 = self.text_feature_injector(_c3_1)
        _c3_2 = self.text_feature_injector(_c3_2)
        diff_feat_c3 = self.wfd_c3([_c3_1, _c3_2]) + F.interpolate(diff_feat_c4, scale_factor=2, mode="bilinear")
        diff_feat_c3 = self.sgse([resize(diff_s, size=diff_feat_c3.size()[2:], mode='bilinear', align_corners=False),diff_feat_c3])
        p_bcd_c3  = self.make_pred_c3_bcd(diff_feat_c3)
        
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        diff_feat_c2 = self.wfd_c2([_c2_1, _c2_2]) + F.interpolate(diff_feat_c3, scale_factor=2, mode="bilinear")
        diff_feat_c2 = self.sgse([resize(diff_s, size=diff_feat_c2.size()[2:], mode='bilinear', align_corners=False),diff_feat_c2])
        p_bcd_c2  = self.make_pred_c2_bcd(diff_feat_c2)
        
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        diff_feat_c1 = self.wfd_c1([_c1_1, _c1_2]) + F.interpolate(diff_feat_c2, scale_factor=2, mode="bilinear")
        diff_feat_c1 = self.sgse([resize(diff_s, size=diff_feat_c1.size()[2:], mode='bilinear', align_corners=False),diff_feat_c1 ])
        p_bcd_c1  = self.make_pred_c1_bcd(diff_feat_c1)

        diff_feat_c4_up = resize(diff_feat_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        diff_feat_c3_up = resize(diff_feat_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        diff_feat_c2_up = resize(diff_feat_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        
        semantic_diff_up = resize(semantic_diff, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        _c_bcd = self.linear_fuse_bcd(torch.cat((diff_feat_c4_up, diff_feat_c3_up, diff_feat_c2_up, diff_feat_c1, semantic_diff_up), dim=1))
        
        x = self.dense_2x(_c_bcd)
        x = self.dense_1x(x)
        p_bcd = self.make_pred_bcd(x)
        
        p_1_refined = self.cgsa_p1(p_1, x)
        p_2_refined = self.cgsa_p2(p_2, x)
        
        outputs.append(p_1_refined)
        outputs.append(p_2_refined)
        outputs.append(p_bcd_c4)
        outputs.append(p_bcd_c3)
        outputs.append(p_bcd_c2)
        outputs.append(p_bcd_c1)
        outputs.append(p_bcd_s)
        outputs.append(p_bcd)  

        return outputs

class TSFDNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256,
                 class_names=None, img_size=512,
                 clip_head_path='/home/zhj/TSFDNet/pretrained/RemoteCLIP-ViT-B-32.pt'):
        super(TSFDNet, self).__init__()
        class_names = require_class_names(class_names)
        self.embed_dims = [64, 128, 320, 512]
        self.depths     = [3, 4, 6, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.
        self.attn_drop = 0.1
        self.drop_path_rate = 0.3 
        self.backbone = Backbone( patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                 num_heads = [1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=self.drop_rate,
                 attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=self.depths, sr_ratios=[8, 4, 2, 1])

        self.decoder = TSFDDecoder(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[2, 4, 8, 16],
                    class_names=class_names, img_size=img_size, clip_head_path=clip_head_path)

    def forward(self, x1, x2):
        x_size = x1.size() 
        [fx1, fx2] = [self.backbone(x1), self.backbone(x2)]
        cp = self.decoder(fx1, fx2)
        
        upsampled_outputs = []
        for output in cp:
            upsampled_outputs.append(F.interpolate(output, size=x_size[2:], mode='bilinear', align_corners=False))

        p1 = upsampled_outputs[0]
        p2 = upsampled_outputs[1]
        final_change = upsampled_outputs[-1]
        aux_changes = upsampled_outputs[2:-1]

        if self.training:
            return [final_change] + aux_changes, p1, p2
        else:
            return final_change, p1, p2

