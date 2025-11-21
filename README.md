# LFINet: Linguistic Semantics Fused Chain-of-Thought Diffusion Model for Realistic Tree Sample Generation from Aerial Imagery
![](Figures/showFig.png)

LFINet is a text-to-image pipeline designed for forestry applications.  
It combines:

- **CLIP-style image–text encoders**
- **Chain-of-Thought (CoT) GRU reasoning**
- **Text-embedded Diffusion Model**

to generate realistic forestry imagery from natural-language descriptions.

### Train LFINet
This stage including:
  * CLIP pre-traing
    - configuration
    ```arduino
    config/clip_forest.json
    ```
    - trainng
    ```bash
     python tools.tran_clip --config config/clip_forest.json
     ```
  * CoT-Diffusion trianing <br>
    - configuration
    ```arduino
    config/diffusion_cot.json
    ```
    - training
     ```bash
     python tools.tran_diffusion_cot --config config/diffusion_cot.json
     ```
### Text-to-Image Sampling
* configuration
```arduino
config/diffusion_cot_sample.json
```
Example:
```json
{
  "prompts": [
    "Aerial view of a dense conifer forest with dark-green clustered crowns."
  ]
}
```
* Run sampling
```bash
 python tools/sample_diffusion_cot.py --config config/diffusion_cot_sample.json
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
