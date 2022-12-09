import numpy as np
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, EulerDiscreteScheduler, \
    StableDiffusionInpaintPipelineLegacy, DDIMPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
from scipy.ndimage import gaussian_filter


class StableSpaceship:
    def __init__(self):
        # new stable diffusion inpaint models encode both img and mask and concat them in the latent;
        #   the old v1-4 (legacy inpaint) starts off with the init img and only allows edits in the mask;
        #   when combined in the latent w/ the black background init img, v2 creates simple ass looking spaceships;
        # models:
        #  legacy:  # diffusers==0.9.0, transformers==4.24.0
        #   'stabilityai/stable-diffusion-2'  # 768x768
        #   'stabilityai/stable-diffusion-2-base'  # 512x512
        #   'CompVis/stable-diffusion-v1-4'  # 512x512
        #   'runwayml/stable-diffusion-v1-5'
        #  new inpainting:
        #   'stabilityai/stable-diffusion-2-inpainting'
        #   'runwayml/stable-diffusion-inpainting'
        self.model_id = 'stabilityai/stable-diffusion-2-base'
        self.device = 'cuda'

        # StableDiffusionInpaintPipeline
        self.pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(self.model_id, revision='fp16', torch_dtype=torch.float16)
        # self.pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, revision='fp16')
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

    def example_run(self, prompt='sideview of a 3D metallic combat spaceship from star trek in space | deviantart',
                    negative_prompt='ugly cartoonish simple', mask_path='spaceship_mask.png', num_inference_steps=50, gen_space_bg=False):
        torch.random.manual_seed(42)
        np.random.seed(42)

        size = (768, 768) if self.model_id == 'stabilityai/stable-diffusion-2' else (512, 512)
        upscale_rnd = 4
        inv_star_dens = 1000

        mask_img = resize((plt.imread(mask_path) > 0.5).astype(np.float32), size).max(axis=-1)

        rnd_spacheship = np.random.uniform(0.1, 0.4, (size[0] // upscale_rnd, size[1] // upscale_rnd, 3))
        rnd_spacheship = resize(rnd_spacheship, size, order=0)
        rnd_spacheship[mask_img <= 0.5] = 0.

        # TODO more efficient stars gen
        space_bg = np.zeros_like(rnd_spacheship)
        if gen_space_bg:
            stars_y = np.random.choice(space_bg.shape[0], int(np.mean(size) // inv_star_dens), replace=False)
            stars_x = np.random.choice(space_bg.shape[1], int(np.mean(size) // inv_star_dens), replace=False)
            stars_s = np.clip(np.random.normal(3, 4, stars_y.shape[0]), 2, 7).astype(int)  # size
            for y, x, s in zip(stars_y, stars_x, stars_s):
                space_bg[y - s // 2:y + s // 2, x - s // 2:x + s // 2] = 1.

            space_bg = gaussian_filter(space_bg, 1.5)
            space_bg[mask_img > 0.5] = 0.

        init_img = rnd_spacheship + space_bg

        init_img_pil = Image.fromarray((init_img * 255).astype(np.uint8))
        mask_img_pil = Image.fromarray((mask_img * 255).astype(np.uint8))

        # from: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion_2
        # import PIL, requests, io
        # url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Starsinthesky.jpg/1280px-Starsinthesky.jpg'
        # dl = lambda url: PIL.Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")
        # init_img_pil = dl(url).resize((512, 512))  # for not legacy inpaints, some bg inspiration for the model

        init_img_param_name = 'init_image' if isinstance(self.pipe, StableDiffusionInpaintPipelineLegacy) else 'image'
        kwargs = {init_img_param_name: init_img_pil}

        with autocast("cuda"):
            images = self.pipe(prompt, mask_image=mask_img_pil, negative_prompt=negative_prompt,
                               height=size[0], width=size[1], guidance_scale=5.,
                               num_inference_steps=num_inference_steps, **kwargs).images

        for img in images:
            plt.figure()
            plt.imshow(np.concatenate([init_img,
                                       np.tile(mask_img[..., None], (1, 1, 3)),
                                       np.asarray(img).astype(np.float32) / 255.], axis=1))
            plt.axis('off')
            plt.tight_layout()
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
        plt.show()

    def run(self, mask_img, prompt='side view of a spaceship', init_img=None, ship_brightness=.3,
            steps=200, guidance_scale=7.5, height=512, width=512):

        # TODO take everything from example run here !!!

        # TODO add negative_prompt
        # TODO add starts to bg so the rest is always painted

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
