from typing import List
from PIL import Image
from controlnet_aux.processor import Processor


class ControlnetAuxService:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.processors = {}
        pass

    def load(self, processors: List[str]):
        for name in processors:
            if name not in self.processors:
                self.processors[name] = Processor(name)

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image = None,
        processors: List[str] = [],
        keep_size: bool = True,
    ):
        if not processors:
            return []

        self.load(processors)

        if mask:
            background = Image.new('RGBA', image.size)
            mask = mask.convert('L')
            image = Image.composite(image, background, mask)

        images = []
        for model in self.processors.values():
            result = model(image, to_pil=True)
            if keep_size:
                result = result.resize(image.size)
            images.append(result)
        return images
