import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)



class CoTGRUReasoner(nn.Module):
    def __init__(self, text_encoder, tokenizer,
                 embed_dim=256,
                 hidden_dim=256,
                 max_len=64):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.gru = nn.GRUCell(embed_dim * 2, hidden_dim)
        self.proj = nn.Linear(hidden_dim, embed_dim)

        # predix
        self.stage_prefixes = [
            "species-level identity of this tree: ",
            "morphological structure of this tree crown: ",
            "phenological or seasonal context of this tree: ",
            "fine-grained textural and branching traits of this tree: ",
        ]

    def _encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        device = next(self.text_encoder.parameters()).device

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            emb = self.text_encoder(ids, mask)   # [B, D]
        return emb

    def forward(self, captions: List[str]) -> torch.Tensor:
        device = next(self.text_encoder.parameters()).device

        base_emb = self._encode_text_batch(captions)  # [B, D]
        B, _ = base_emb.shape

        h = torch.zeros(B, self.hidden_dim, device=device)

        for prefix in self.stage_prefixes:
            stage_texts = [prefix + c for c in captions]
            stage_emb = self._encode_text_batch(stage_texts)  # [B, D]
            inp = torch.cat([base_emb, stage_emb], dim=-1)    # [B, 2D]
            h = self.gru(inp, h)

        return self.proj(h)  # [B, embed_dim]



class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim):
        super().__init__()


        self.norm1 = nn.GroupNorm(32, in_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act = nn.SiLU()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.temb_proj = nn.Linear(temb_dim, out_ch)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, temb):
        # x: [B, C, H, W]
        # temb: [B, temb_dim]
        h = self.conv1(self.act(self.norm1(x)))

        t = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        h = h + t

        h = self.conv2(self.act(self.norm2(h)))
        return self.skip(x) + h


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, down=True):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, temb_dim)
        self.down = down
        if down:
            self.downsample = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

    def forward(self, x, temb):
        h = self.res(x, temb)
        skip = h
        if self.down:
            h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, temb_dim, up=True):
        super().__init__()
        self.up = up
        if up:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1)
        self.res = ResBlock(in_ch + skip_ch, out_ch, temb_dim)

    def forward(self, x, skip, temb):
        if self.up:
            x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, temb)



class CoTUNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base_ch=64,
        ch_mult=(1, 2, 4, 8),
        temb_dim=256,
        cond_dim=256,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch

        # 3 -> 64
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(temb_dim),
            nn.Linear(temb_dim, temb_dim * 4),
            nn.SiLU(),
            nn.Linear(temb_dim * 4, temb_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        ch = [base_ch * m for m in ch_mult]        # [64,128,256,512]
        in_channels  = [ch[0]] + ch[:-1]           # [64,64,128,256]
        out_channels = ch                          # [64,128,256,512]

        # ---------- Down blocks ----------
        self.down_blocks = nn.ModuleList()
        for i in range(len(ch)):
            self.down_blocks.append(
                DownBlock(
                    in_ch=in_channels[i],
                    out_ch=out_channels[i],
                    temb_dim=temb_dim,
                    down=(i != len(ch) - 1)       
                )
            )

        mid_ch = out_channels[-1]                 # 512

        # ---------- Middle ----------
        self.mid1 = ResBlock(mid_ch, mid_ch, temb_dim)
        self.mid2 = ResBlock(mid_ch, mid_ch, temb_dim)

        # ---------- Up blocks ----------
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(ch) - 1)):     # i = 2,1,0
            in_ch_up = ch[i + 1]                  # 512,256,128
            skip_ch  = ch[i]                      # 256,128,64
            out_ch_up = ch[i]                     # 256,128,64
            self.up_blocks.append(
                UpBlock(
                    in_ch=in_ch_up,
                    skip_ch=skip_ch,
                    out_ch=out_ch_up,
                    temb_dim=temb_dim,
                    up=True
                )
            )

        self.norm_out = nn.GroupNorm(32, base_ch)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t, cond_emb):
        # t: [B], cond_emb: [B, D]
        temb = self.time_mlp(t) + self.cond_mlp(cond_emb)

        h = self.conv_in(x)
        skips = []


        for i, blk in enumerate(self.down_blocks):
            h, skip = blk(h, temb)
            if i != len(self.down_blocks) - 1:   
                skips.append(skip)

        # bottleneck
        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        for blk in self.up_blocks:
            skip = skips.pop()                  
            h = blk(h, skip, temb)

        h = self.conv_out(self.act(self.norm_out(h)))
        return h
