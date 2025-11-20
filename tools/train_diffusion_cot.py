import os
import json
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.forest_dataset import ForestCaptionDataset
from models.clip_model import TextTransformerEncoder
from models.cot_unet import CoTGRUReasoner, CoTUNet
import argparse



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion CoT")
    parser.add_argument(
        "--config",
        type=str,
        default="config/diffusion_cot.json",
        help="Path to JSON config file",
    )
    return parser.parse_args()


class LinearBetaScheduler:
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)

        betas = betas.to(device)
        alphas = alphas.to(device)
        acp = acp.to(device)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = acp

        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), acp[:-1]]
        )

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return alphas_cumprod_t.sqrt() * x0 + (1 - alphas_cumprod_t).sqrt() * noise, noise




def main():
    args = parse_args()
    cfg = load_json(args.config)

    seed = cfg["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ---- Load Dataset ---- #
    ds_cfg = cfg["dataset"]
    dataset = ForestCaptionDataset(
        im_path=ds_cfg["root"],
        im_size=ds_cfg["im_size"],
        im_channels=ds_cfg["im_channels"],
        im_ext="jpg",
        use_latents=False,
        caption_key=ds_cfg["caption_keys"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=ds_cfg["batch_size"],
        shuffle=True,
        num_workers=ds_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # ---- Load text tokenizer & encoder ---- #
    clip_cfg = cfg["clip_model"]
    tokenizer = AutoTokenizer.from_pretrained(clip_cfg["tokenizer_name"])

    txt_cfg = clip_cfg["text_encoder"]
    text_encoder = TextTransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=txt_cfg["embed_dim"],
        max_len=txt_cfg["max_len"],
        depth=txt_cfg["depth"],
        num_heads=txt_cfg["num_heads"],
    ).to(device)

    ckpt_path = os.path.join(cfg["paths"]["clip_ckpt_dir"], "text_encoder_best.pt")
    print("Loading text encoder:", ckpt_path)
    text_encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    # ---- Build CoT ---- #
    cot_cfg = cfg["cot"]
    cot_reasoner = CoTGRUReasoner(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        embed_dim=cot_cfg["embed_dim"],
        hidden_dim=cot_cfg["hidden_dim"],
        max_len=cfg["train"]["max_text_len"],
    ).to(device)

    # ---- Build U-Net ---- #
    unet_cfg = cfg["unet"]
    unet = CoTUNet(
        in_ch=ds_cfg["im_channels"],
        base_ch=unet_cfg["base_ch"],
        ch_mult=tuple(unet_cfg["ch_mult"]),
        temb_dim=cot_cfg["embed_dim"],
        cond_dim=cot_cfg["embed_dim"]
    ).to(device)

    # ---- Optimizer ---- #
    optimizer = torch.optim.AdamW(
        list(cot_reasoner.parameters()) + list(unet.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # ---- Diffusion Scheduler ---- #
    diff_cfg = cfg["diffusion"]
    scheduler = LinearBetaScheduler(
        timesteps=diff_cfg["timesteps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        device=device,
    )

    # ---- Training Loop ---- #
    num_epochs = cfg["train"]["epochs"]
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    global_step = 0

    for epoch in range(num_epochs):
        unet.train()
        cot_reasoner.train()

        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, cond in pbar:
            images = images.to(device)
            captions = cond["text"]

            cond_emb = cot_reasoner(captions)

            B = images.size(0)
            T = diff_cfg["timesteps"]
            t = torch.randint(0, T, (B,), device=device).long()

            x_t, noise = scheduler.q_sample(images, t)
            noise_pred = unet(x_t, t, cond_emb)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{epoch_loss/(global_step+1):.4f}"})

            global_step += 1


        torch.save(
            unet.state_dict(),
            os.path.join(out_dir, f"unet_epoch{epoch+1}.pt")
        )
        torch.save(
            cot_reasoner.state_dict(),
            os.path.join(out_dir, f"cot_epoch{epoch+1}.pt")
        )

        print(f"[Epoch {epoch+1}] Loss = {epoch_loss/(global_step+1):.4f}")


if __name__ == "__main__":
    main()
