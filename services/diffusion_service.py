import os
import torch
import random
from compel import Compel
from typing import Dict, List
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    MultiControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
)


class DiffusionService:
    def __init__(self, device: str = 'cuda'):
        self.pipe = None
        self.ldm_path: str = ''
        self.vae_path: str = ''
        self.compel = None
        self.cnet_paths: List = []
        self.cnet_dicts: Dict = {}
        self.device = device

    def load(self, ldm_path: str, vae_path: str):
        if self.ldm_path == ldm_path and self.vae_path == vae_path:
            return

        if not self.pipe:
            del self.pipe

        self.pipe = StableDiffusionPipeline.from_single_file(
            ldm_path,
            vae=AutoencoderKL.from_single_file(vae_path),
            safety_checker=None,
        )
        self.compel = Compel(self.pipe.tokenizer, self.pipe.text_encoder)
        self.ldm_path = ldm_path
        self.vae_path = vae_path

    def get_controlnet_list(self, cnet_paths: List[str]):
        if self.cnet_paths == cnet_paths:
            return

        models = []
        for name in cnet_paths:
            if name not in self.cnet_dicts:
                model = ControlNetModel.from_pretrained(
                    name,
                    torch_dtype=torch.float16,
                )
                self.cnet_dicts[name] = model
            models.append(self.cnet_dicts[name])
        return MultiControlNetModel(models)

    def create_generator(self, seed: int):
        seed = random.randint(11111, 999999999) if seed < 1 else seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return generator

    def create_prompt_embeds(self, prompt: str, negative_prompt: str):
        prompt_embeds = self.compel(prompt)
        negative_prompt_embeds = self.compel(negative_prompt)
        return prompt_embeds, negative_prompt_embeds

    def txt2img(self, **kwargs):
        return self.pipe(**kwargs).images

    def img2img(self, **kwargs):
        return StableDiffusionImg2ImgPipeline.from_pipe(self.pipe)(**kwargs)

    def inpaint(self, **kwargs):
        return StableDiffusionInpaintPipeline.from_pipe(self.pipe)(**kwargs)

    def cnet_txt2img(
        self,
        control_paths: List[str],
        **kwargs,
    ):
        pipe = StableDiffusionControlNetPipeline.from_pipe(
            self.pipe,
            controlnet=self.get_controlnet_list(control_paths),
        )
        return pipe(**kwargs)

    def cnet_img2img(
        self,
        control_paths: List[str],
        **kwargs,
    ):
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pipe(
            self.pipe,
            controlnet=self.get_controlnet_list(control_paths),
        )
        return pipe(**kwargs)

    def cnet_inpaint(
        self,
        control_paths: List[str],
        **kwargs,
    ):
        pipe = StableDiffusionControlNetInpaintPipeline.from_pipe(
            self.pipe,
            controlnet=self.get_controlnet_list(control_paths),
        )
        return pipe(**kwargs)
