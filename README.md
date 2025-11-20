# LFINet: Linguistic Semantics Fused Chain-of-Thought Diffusion Model for Realistic Tree Sample Generation from Aerial Imagery

LFINet is a text-to-image pipeline designed for forestry applications.  
It combines:

- **CLIP-style image–text encoders**
- **Chain-of-Thought (CoT) GRU reasoning**
- **Text-embedded Diffusion Model**

to generate realistic forestry imagery from natural-language descriptions.


### Train LFINet
This stage trains:
  * CLIP pre-traing <br>
    ```bash python tools.tran_clip --config config/clip_forest.json ```
```json
clip_forest.json
{
  "seed": 42,
  "output_dir": "runs/clip_forest",

  "dataset": {
    "root": "../data/image_text",
    "im_size": 128,
    "im_channels": 3,
    "caption_keys": ["caption", "text"]
  },

  "train": {
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "max_text_len": 64,
    "log_interval": 50,
    "save_every": 1
  },

  "model": {
    "embed_dim": 512,

    "img_encoder": {
      "img_size": 128,
      "patch_size": 8,
      "in_chans": 3,
      "embed_dim": 512,
      "depth": 6,
      "num_heads": 8
    },

    "txt_encoder": {
      "embed_dim": 512,
      "max_len": 64,
      "depth": 6,
      "num_heads": 8
    },

    "tokenizer_name": "bert-base-uncased"
  }
}
```
  * CoT-Diffusion trianing <br>
```bash python tools.tran_diffusion_cot --config config/diffusion_cot.json ```<br>
```json
diffusion_cot.json
{
  "seed": 42,

  "dataset": {
    "root": "../data/image_text",
    "im_size": 128,
    "im_channels": 3,
    "caption_keys": ["caption", "text"],
    "batch_size": 16,
    "num_workers": 4
  },

  "diffusion": {
    "timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02
  },

  "train": {
    "epochs": 20,
    "lr": 0.0001,
    "weight_decay": 0.01,
    "max_text_len": 64,
    "log_interval": 50,
    "save_every": 1
  },

  "cot": {
    "embed_dim": 512,
    "hidden_dim": 1024
  },

  "unet": {
    "base_ch": 64,
    "ch_mult": [1, 2, 4, 8],
    "img_size": 128
  },

  "clip_model": {
    "tokenizer_name": "bert-base-uncased",
    "text_encoder": {
      "embed_dim": 512,
      "max_len": 64,
      "depth": 6,
      "num_heads": 8
    }
  },

  "paths": {
    "clip_ckpt_dir": "runs/clip_forest",
    "output_dir": "runs/diffusion_cot"
  }
}

```
### Dataset Format
```bash
  data/image_text/
  ├── 0.jpg
  ├── 0.json
  ├── 1.jpg
  ├── 1.json
  ├── 2.jpg
  ├── 2.json
  └── ...
```
Each image and JSON have the same prefix (e.g., 1.jpg + 1.json)
<br>Dataset download [here](https://doi.org/10.5281/zenodo.17664421)
