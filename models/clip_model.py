import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageTransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=8,
        in_chans=3,
        embed_dim=256,
        depth=6,
        num_heads=8,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,        # 方便使用 [B, N, C]
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: [B, 3, H, W] 
        input:
          latent_map: [B, embed_dim, H_p, W_p]
          global_emb: [B, embed_dim]
        """
        B, C, H, W = x.shape
        # patch embedding: [B, embed_dim, H_p, W_p]
        feat = self.patch_embed(x)
        H_p, W_p = feat.shape[-2:]
        N = H_p * W_p

        # [B, C, H_p, W_p] -> [B, N, C]
        tokens = feat.flatten(2).transpose(1, 2)   # [B, N, embed_dim]

        tokens = tokens + self.pos_embed[:, :N, :]

        tokens = self.encoder(tokens)  # [B, N, D]

        global_emb = tokens.mean(dim=1)   # [B, D]

        latent_map = tokens.transpose(1, 2).reshape(B, self.embed_dim, H_p, W_p)

        return latent_map, global_emb


class TextTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        max_len=64,
        depth=4,
        num_heads=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, L]
        attention_mask: [B, L] (1 for valid, 0 for pad) 或 None
        """
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_embed(input_ids) + self.pos_embed(pos)  # [B, L, D]

        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # [B, L], True for pad
        else:
            key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, D]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            x = x.mean(dim=1)

        return x  # [B, D]

class CLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=256):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, images, input_ids, attention_mask=None):

        latent_map, img_emb = self.image_encoder(images)  # [B,D,H',W'], [B,D]

        txt_emb = self.text_encoder(input_ids, attention_mask)  # [B,D]

        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        logit_scale = self.logit_scale.exp()

        # 图像对文本的相似度矩阵：[B,B]
        logits_per_image = logit_scale * img_emb @ txt_emb.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, latent_map, img_emb, txt_emb




