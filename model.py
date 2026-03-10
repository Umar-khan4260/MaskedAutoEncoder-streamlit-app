"""
model.py — MAE Model Architecture (pure PyTorch, no pretrained libraries)
Extracted from the training notebook for standalone inference use.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Patch Utilities ───────────────────────────────────────────────────────────

def patchify(imgs, patch_size=16):
    B, C, H, W = imgs.shape
    p = patch_size
    h_patches = H // p
    w_patches = W // p
    x = imgs.reshape(B, C, h_patches, p, w_patches, p)
    x = x.permute(0, 2, 4, 3, 5, 1)
    patches = x.reshape(B, h_patches * w_patches, p * p * C)
    return patches


def unpatchify(patches, patch_size=16, img_size=224):
    p = patch_size
    h = w = img_size // p
    B = patches.shape[0]
    x = patches.reshape(B, h, w, p, p, 3)
    x = x.permute(0, 5, 1, 3, 2, 4)
    imgs = x.reshape(B, 3, h * p, w * p)
    return imgs


def random_masking(x, mask_ratio=0.75):
    B, N, D = x.shape
    num_keep = int(N * (1 - mask_ratio))
    noise        = torch.rand(B, N, device=x.device)
    ids_shuffle  = torch.argsort(noise, dim=1)
    ids_restore  = torch.argsort(ids_shuffle, dim=1)
    ids_keep     = ids_shuffle[:, :num_keep]
    x_visible    = torch.gather(
        x, dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )
    mask = torch.ones(B, N, device=x.device)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_visible, mask, ids_restore


# ── Positional Embedding ──────────────────────────────────────────────────────

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid   = np.meshgrid(grid_w, grid_h)
    grid   = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    pos_embed = _get_2d_sincos_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed


def _get_2d_sincos_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega  = 1.0 / (10000 ** omega)
    pos    = pos.reshape(-1)
    out    = np.outer(pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ── Transformer Building Blocks ───────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=True)
        self.proj      = nn.Linear(dim, dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ── Encoder: ViT-Base/16 ──────────────────────────────────────────────────────

class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        pos_embed = get_2d_sincos_pos_embed(embed_dim, img_size // patch_size)
        self.register_buffer(
            'pos_embed',
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(
            self.patch_embed.weight.view(self.patch_embed.weight.size(0), -1)
        )
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, imgs, mask_ratio=0.75):
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x, mask, ids_restore = random_masking(x, mask_ratio)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore


# ── Decoder: ViT-Small/16 ─────────────────────────────────────────────────────

class MAEDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, encoder_dim=768,
                 decoder_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch_size  = patch_size
        self.decoder_dim = decoder_dim
        patch_dim        = patch_size * patch_size * 3

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        pos_embed = get_2d_sincos_pos_embed(decoder_dim, img_size // patch_size)
        self.register_buffer(
            'pos_embed',
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm      = nn.LayerNorm(decoder_dim)
        self.pred_head = nn.Linear(decoder_dim, patch_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, latent, ids_restore):
        B   = latent.shape[0]
        N   = ids_restore.shape[1]
        N_vis = latent.shape[1]

        x         = self.enc_to_dec(latent)
        n_masked  = N - N_vis
        mask_tokens = self.mask_token.expand(B, n_masked, -1)

        x_full  = torch.cat([x, mask_tokens], dim=1)
        ids_exp = ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        x_full  = torch.gather(x_full, dim=1, index=ids_exp)
        x_full  = x_full + self.pos_embed

        for block in self.blocks:
            x_full = block(x_full)

        return self.pred_head(self.norm(x_full))


# ── Full MAE Model ────────────────────────────────────────────────────────────

class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 enc_dim=768, enc_depth=12, enc_heads=12,
                 dec_dim=384, dec_depth=12, dec_heads=6,
                 mlp_ratio=4.0, mask_ratio=0.75, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.img_size   = img_size

        self.encoder = MAEEncoder(
            img_size=img_size, patch_size=patch_size,
            embed_dim=enc_dim, depth=enc_depth,
            num_heads=enc_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.decoder = MAEDecoder(
            img_size=img_size, patch_size=patch_size,
            encoder_dim=enc_dim, decoder_dim=dec_dim,
            depth=dec_depth, num_heads=dec_heads,
            mlp_ratio=mlp_ratio, dropout=dropout
        )

    def forward(self, imgs, mask_ratio=None):
        ratio              = mask_ratio or self.mask_ratio
        latent, mask, ids  = self.encoder(imgs, ratio)
        pred               = self.decoder(latent, ids)
        target             = patchify(imgs, self.patch_size)
        mean               = target.mean(dim=-1, keepdim=True)
        var                = target.var(dim=-1, keepdim=True)
        target_norm        = (target - mean) / (var + 1e-6).sqrt()
        loss_per_patch     = ((pred - target_norm) ** 2).mean(dim=-1)
        loss               = (loss_per_patch * mask).sum() / mask.sum()
        return loss, pred, mask

    @torch.no_grad()
    def reconstruct(self, imgs, mask_ratio=None):
        """
        Returns: masked_img, recon_img, mask
        All tensors in the same shape as imgs: (B, 3, H, W)
        """
        self.eval()
        ratio             = mask_ratio or self.mask_ratio
        _, pred, mask     = self.forward(imgs, ratio)

        target      = patchify(imgs, self.patch_size)
        mean        = target.mean(dim=-1, keepdim=True)
        var         = target.var(dim=-1, keepdim=True)
        pred_pixels = pred * (var + 1e-6).sqrt() + mean

        recon_img = unpatchify(pred_pixels, self.patch_size, self.img_size)

        mask_exp      = mask.unsqueeze(-1).expand_as(target)
        gray          = 0.5 * torch.ones_like(target)
        masked_patches = target * (1 - mask_exp) + gray * mask_exp
        masked_img     = unpatchify(masked_patches, self.patch_size, self.img_size)

        return masked_img, recon_img, mask


def load_model(checkpoint_path: str, device: str = 'cpu') -> MaskedAutoencoder:
    """
    Load a trained MAE model from a checkpoint file.
    Handles both DataParallel-wrapped and plain state dicts.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and our checkpoint dict format
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    # Strip 'module.' prefix if saved from DataParallel
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model = MaskedAutoencoder()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
