import os
import json
import torch
import torchvision.utils as vutils
from transformers import AutoTokenizer
from models.cot_unet import CoTUNet, CoTGRUReasoner
from models.clip_model import TextTransformerEncoder  
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion CoT")
    parser.add_argument(
        "--config",
        type=str,
        default="config/diffusion_cot_sample.json",
        help="Path to JSON config file",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class LinearBetaScheduler:
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)                   # [T]
        self.alphas = alphas.to(device)                 # [T]
        self.alphas_cumprod = alphas_cumprod.to(device) # [T]

        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )

        # posterior variance β̃_t
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )


@torch.no_grad()
def sample_from_text(
    unet,
    reasoner,
    prompts,
    scheduler,
    img_size=128,
    device="cuda",
    out_dir="samples_cot",
    prefix="sample",
):
    """
    prompts: List[str]
    """
    unet.eval()
    reasoner.eval()
    os.makedirs(out_dir, exist_ok=True)

    B = len(prompts)

    # x_T ~ N(0, I)
    x_t = torch.randn(B, 3, img_size, img_size, device=device)

   
    cond_emb = reasoner(prompts)

    T = scheduler.timesteps

    for step in reversed(range(T)):
        t = torch.full((B,), step, device=device, dtype=torch.long)

        #  ε_theta(x_t, t, cond)
        eps = unet(x_t, t, cond_emb)

        # beta_t = scheduler.betas[step]
        alpha_t = scheduler.alphas[step]
        alpha_bar_t = scheduler.alphas_cumprod[step]
        # alpha_bar_prev = scheduler.alphas_cumprod_prev[step]
        posterior_var = scheduler.posterior_variance[step]


        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)

        mean = coef1 * (x_t - coef2 * eps)

        if step > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(posterior_var) * noise
        else:
            x_t = mean

    # [-1, 1] -> [0, 1]->[0,256]
    x_0 = x_t.clamp(-1.0, 1.0)
    x_0 = (x_0 + 1.0) / 2.0

    for i in range(B):
        save_path = os.path.join(out_dir, f"{prefix}_{i}.png")
        vutils.save_image(x_0[i], save_path)
        print(f"[+] saving image: {save_path}")



def main():

    args = parse_args()
    cfg = load_json(args.config)

    if cfg.get("device", "auto") == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["device"]

    text_cfg = cfg["text"]
    clip_cfg = cfg["clip_model"]
    unet_cfg = cfg["unet"]
    diff_cfg = cfg["diffusion"]
    samp_cfg = cfg["sampling"]

    # 3) tokenizer
    tokenizer_name = text_cfg["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=True
    )


    text_encoder = TextTransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=clip_cfg["embed_dim"],
        max_len=text_cfg["max_len"],
        depth=clip_cfg["txt_depth"],
        num_heads=clip_cfg["txt_heads"],
    ).to(device)

    text_encoder_ckpt = text_cfg["text_encoder_ckpt"]
    state = torch.load(text_encoder_ckpt, map_location=device)
    text_encoder.load_state_dict(state)
    text_encoder.eval()

    reasoner = CoTGRUReasoner(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        embed_dim=clip_cfg["embed_dim"],
        hidden_dim=clip_cfg["embed_dim"],
        max_len=text_cfg["max_len"],
    ).to(device)

    reasoner_ckpt = text_cfg["reasoner_ckpt"]
    reasoner.load_state_dict(torch.load(reasoner_ckpt, map_location=device))
    reasoner.eval()

    unet = CoTUNet(
        in_ch=unet_cfg["in_ch"],
        base_ch=unet_cfg["base_ch"],
        ch_mult=tuple(unet_cfg["ch_mult"]),
        temb_dim=unet_cfg["temb_dim"],
        cond_dim=clip_cfg["embed_dim"]
    ).to(device)

    unet_ckpt = unet_cfg["unet_ckpt"]
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
    unet.eval()

    scheduler = LinearBetaScheduler(
        timesteps=diff_cfg["timesteps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        device=device,
    )

    prompts = samp_cfg["prompts"]
    print("[Info] using prompts：")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p}")

    sample_from_text(
        unet=unet,
        reasoner=reasoner,
        prompts=prompts,
        scheduler=scheduler,
        img_size=unet_cfg["img_size"],
        device=device,
        out_dir=diff_cfg.get("sample_dir", "samples_cot"),
        prefix="forest",
    )


if __name__ == "__main__":
    main()
