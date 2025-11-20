# LFINet: Linguistic Semantics Fused Chain-of-Thought Diffusion Model for Realistic Tree Sample Generation from Aerial Imagery

LFINet is a text-to-image pipeline designed for forestry applications.  
It combines:

- **CLIP-style image–text encoders**
- **Chain-of-Thought (CoT) GRU reasoning**
- **Text-embedded Diffusion Model**

to generate realistic forestry imagery from natural-language descriptions.

### Train LFINet
This stage including:
  * CLIP pre-traing <br>
    ```bash
     python tools.tran_clip --config config/clip_forest.json
     ```
  * CoT-Diffusion trianing <br>
     ```bash
     python tools.tran_diffusion_cot --config config/diffusion_cot.json
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
