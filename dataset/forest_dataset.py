import glob
import os
import json
import torchvision
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data import Dataset


class ForestCaptionDataset(Dataset):
    r"""
    Dataset for directory:
        0.jpg, 0.json, 1.jpg, 1.json, ...

    Each JSON file should contain at least a "caption" field.
    The dataset returns a 128x128 image tensor in [-1, 1] and
    the corresponding condition text (caption).
    """

    def __init__(self,
                 im_path,
                 im_size=128,
                 im_channels=3,
                 im_ext='jpg',
                 use_latents=False,
                 latent_path=None,
                 caption_key=["caption","text"]):
        """
        im_path:    directory containing N.jpg and N.json
        im_size:    output image size (128)
        im_channels: number of image channels (3)
        im_ext:     image extension to look for ('jpg' by default)
        use_latents: if True, load latents from latent_path
        latent_path: npz / pth with precomputed latents
        caption_key: key in json used as condition text
        """
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.caption_key = caption_key

        self.images, self.texts = self.load_images_and_texts(im_path)

        self.use_latents = False
        self.latent_maps = None
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found or length mismatch')

        # 统一的图像
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor(),   # [0,1]
        ])

    def load_images_and_texts(self, im_path):

        assert os.path.exists(im_path), f"images path {im_path} does not exist"

        pattern = os.path.join(im_path, f'*.{self.im_ext}')
        fnames = sorted(glob.glob(pattern))
        images = []
        texts = []

        for fname in tqdm(fnames, desc="Loading image paths"):
            images.append(fname)
            base = os.path.splitext(os.path.basename(fname))[0]
            json_path = os.path.join(im_path, f"{base}.json")

            caption = ""
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(self.caption_key, (list, tuple)):
                        for key in self.caption_key:
                            if key in data and data[key]:
                                caption = data[key]
                                break
                        else:
                            caption = ""
                    else:
                        # caption_key 是单个字符串
                        caption = data.get(self.caption_key, "")               

                except Exception as e:
                    print(f"[warning] read {json_path} failed: {e}")
                    caption = ""
            else:
                print(f"[warning] can not find json: {json_path}")
                caption = ""

            texts.append(caption)
        print(f"Found {len(images)} images with captions")
        return images, texts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        cond_text = self.texts[index]

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            return latent, {"text": cond_text}
        else:
            im_path = self.images[index]
            im = Image.open(im_path).convert("RGB")
            im_tensor = self.transform(im)   # [0,1]
            im.close()

            im_tensor = 2 * im_tensor - 1

            return im_tensor, {"text": cond_text}
