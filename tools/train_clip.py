import os
import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.forest_dataset import ForestCaptionDataset
from models.clip_model import (
    ImageTransformerEncoder,
    TextTransformerEncoder,
    CLIPModel
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP-style Image/Text Transformers")
    parser.add_argument(
        "--config",
        type=str,
        default="config/clip_forest.json",
        help="Path to JSON config file",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def build_dataloader(cfg):
    ds_cfg = cfg["dataset"]

    dataset = ForestCaptionDataset(
        im_path=ds_cfg["root"],
        im_size=ds_cfg["im_size"],
        im_channels=ds_cfg["im_channels"],
        im_ext="jpg",
        use_latents=False,
        caption_key=ds_cfg.get("caption_keys", ["caption", "text"]),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    return dataset, loader




def build_model_and_tokenizer(cfg, device):
    model_cfg = cfg["model"]

    # tokenizer


    tokenizer_name = model_cfg["tokenizer_name"]
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # image encoder
    img_cfg = model_cfg["img_encoder"]
    image_encoder = ImageTransformerEncoder(
        img_size=img_cfg["img_size"],
        patch_size=img_cfg["patch_size"],
        in_chans=img_cfg["in_chans"],
        embed_dim=img_cfg["embed_dim"],
        depth=img_cfg["depth"],
        num_heads=img_cfg["num_heads"],
    )

    # text encoder
    txt_cfg = model_cfg["txt_encoder"]
    text_encoder = TextTransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=txt_cfg["embed_dim"],
        max_len=txt_cfg["max_len"],
        depth=txt_cfg["depth"],
        num_heads=txt_cfg["num_heads"],
    )

    model = CLIPModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        embed_dim=model_cfg["embed_dim"],
    ).to(device)

    return model, tokenizer

def clip_loss(logits_per_image, logits_per_text):
    B = logits_per_image.size(0)
    labels = torch.arange(B, device=logits_per_image.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def train_one_epoch(model, tokenizer, data_loader, optimizer, cfg, device, epoch_idx):
    model.train()
    train_cfg = cfg["train"]
    log_interval = train_cfg["log_interval"]
    max_text_len = train_cfg["max_text_len"]

    running_loss = 0.0
    step = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch_idx + 1}", ncols=100)
    for images, cond in pbar:
        step += 1

        # images: [B,3,128,128], 已经是 [-1,1]
        images = images.to(device, non_blocking=True)

        captions = cond["text"]
        batch_enc = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        input_ids = batch_enc["input_ids"].to(device, non_blocking=True)
        attention_mask = batch_enc["attention_mask"].to(device, non_blocking=True)

        logits_per_image, logits_per_text, latent_map, img_emb, txt_emb = model(
            images, input_ids, attention_mask
        )

        loss = clip_loss(logits_per_image, logits_per_text)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / step

        if step % log_interval == 0:
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return running_loss / step


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(cfg["seed"])

    # data
    _, data_loader = build_dataloader(cfg)

    # model + tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg, device)

    # optimizer
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    best_loss = float("inf")
    num_epochs = train_cfg["epochs"]

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model, tokenizer, data_loader, optimizer, cfg, device, epoch
        )
        print(f"Epoch {epoch + 1}/{num_epochs} - loss: {avg_loss:.4f}")

        # save checkpoints
        out_dir = cfg["output_dir"]
        if (epoch + 1) % train_cfg["save_every"] == 0:
            torch.save(
                model.image_encoder.state_dict(),
                os.path.join(out_dir, f"image_encoder_epoch{epoch + 1}.pt"),
            )
            torch.save(
                model.text_encoder.state_dict(),
                os.path.join(out_dir, f"text_encoder_epoch{epoch + 1}.pt"),
            )
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, f"clip_model_epoch{epoch + 1}.pt"),
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.image_encoder.state_dict(),
                os.path.join(out_dir, "image_encoder_best.pt"),
            )
            torch.save(
                model.text_encoder.state_dict(),
                os.path.join(out_dir, "text_encoder_best.pt"),
            )
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, "clip_model_best.pt"),
            )
            print(f"  ↳ New best model saved (loss={best_loss:.4f})")

    print("Training finished.")


if __name__ == "__main__":
    main()
