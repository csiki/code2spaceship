import numpy as np
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image


# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)


class StableSpaceship:
    def __init__(self):
        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cuda"

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            # revision="fp16",
            # torch_dtype=torch.float16,
            use_auth_token=True  # TODO use model in stable-diffusion-v1-4
        )
        self.pipe = pipe.to(device)

    def example_run(self):
        size = (512, 512)
        mask_img = resize((plt.imread('spaceship_mask.png') > 0.5).astype(np.float32), size).max(axis=-1)
        init_img = np.random.uniform(0, 0.3, (size[0], size[1], 3))  # could become anything
        init_img[mask_img <= 0.5] = 0  # black space

        mask_img = Image.fromarray((mask_img * 255).astype(np.uint8))
        init_img = Image.fromarray((init_img * 255).astype(np.uint8))

        prompt = 'sideview of a spaceship from star wars with silver and blue colors r2d2 on top'
        with autocast("cuda"):
            images = self.pipe(prompt, init_image=init_img, mask_image=mask_img, guidance_scale=7.5,
                               num_inference_steps=200).images

        for img in images:
            plt.imshow(img)
        plt.show()

    def run(self, mask_img, prompt='side view of a spaceship', init_img=None, ship_brightness=.3,
            steps=200, guidance_scale=7.5, height=512, width=512):

        assert height == 512 and width == 512

        mask_img = resize(mask_img, (height, width))

        if init_img is None:
            mask_div = 128 if mask_img.dtype == np.uint8 else .5
            init_img = np.random.uniform(0, ship_brightness, (height, width, 3))  # could become anything
            init_img[mask_img <= mask_div] = 0  # black space
        init_img = Image.fromarray((init_img * 255).astype(np.uint8))

        if mask_img.dtype != np.uint8:
            mask_img = (mask_img * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_img)

        with autocast("cuda"):
            img = self.pipe(prompt, init_image=init_img, mask_image=mask_img, guidance_scale=guidance_scale,
                            num_inference_steps=steps).images[0]

        return np.array(img, dtype=np.float32) / 255.


if __name__ == '__main__':
    sss = StableSpaceship()
    sss.example_run()
