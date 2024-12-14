import os
import math
import requests
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, List, Tuple, Union


def load_image(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, str):
        try:
            if image.startswith("http://") or image.startswith("https://"):
                response = requests.get(image, stream=True)
                response.raise_for_status()  # Raise exception for non-2xx status codes
                image = Image.open(response.raw)
            elif os.path.isfile(image):
                image = Image.open(image)
            else:
                raise ValueError("Incorrect path or URL.")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error downloading image: {e}")
    elif isinstance(image, Image.Image):
        pass  # Image object already loaded, do nothing
    else:
        raise ValueError("Incorrect format used for the image.")
    return image


def resize_and_center_crop(
    image: Union[str, Image.Image],
    max_size: int = 1024,
    div_size: int = 8,
) -> Image.Image:
    image = load_image(image)
    width, height = image.size
    ratio = max(width / max_size, height / max_size)
    width = int(round(width / ratio))
    height = int(round(height / ratio))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    final_width = div_size * round(width / div_size)
    final_height = div_size * round(height / div_size)
    left = (width - final_width) // 2
    top = (height - final_height) // 2
    right = final_width + left
    bottom = final_height + top
    result = resized.crop((left, top, right, bottom))
    return result


def make_image_grid(
    images: List[Image.Image],
    cols: int = 2,
    resize: int = None,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color=background)
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    if resize:
        grid = resize_and_center_crop(grid, resize, 8)
    return grid


def merge_mask(masks: List[Image.Image]) -> Image.Image:
    image = Image.new('L', masks[0].size)
    for mask in masks:
        image = Image.composite(mask, image, mask)
    return image


def mask_dilate(mask: Image.Image, size: int = 3) -> Image.Image:
    if size <= 0:
        return mask
    size = size - 1 if size % 2 == 0 else size
    return mask.filter(ImageFilter.MaxFilter(size))


def mask_blur(mask: Image.Image, size: int = 3) -> Image.Image:
    if size <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(size))


def gradient(
    size: Tuple[int, int],
    mode: str = 'horizontal',
    colors: Dict = None,
    tolerance: int = 0,
) -> Image.Image:
    if colors is None:
        colors = {0: [0, 0, 0], 50: [50, 50, 50], 100: [100, 100, 100]}

    colors = {int(k): [int(c) for c in v] for k, v in colors.items()}
    colors[0] = colors[min(colors.keys())]
    colors[255] = colors[max(colors.keys())]

    img = Image.new('RGB', size, color=(0, 0, 0))

    color_stop_positions = sorted(colors.keys())
    color_stop_count = len(color_stop_positions)
    spectrum = []
    for i in range(256):
        start_pos = max(p for p in color_stop_positions if p <= i)
        end_pos = min(p for p in color_stop_positions if p >= i)
        start = colors[start_pos]
        end = colors[end_pos]

        if start_pos == end_pos:
            factor = 0
        else:
            factor = (i - start_pos) / (end_pos - start_pos)

        r = round(start[0] + (end[0] - start[0]) * factor)
        g = round(start[1] + (end[1] - start[1]) * factor)
        b = round(start[2] + (end[2] - start[2]) * factor)
        spectrum.append((r, g, b))

    draw = ImageDraw.Draw(img)
    if mode == 'horizontal':
        for x in range(size[0]):
            pos = int(x * 100 / (size[0] - 1))
            color = spectrum[pos]
            if tolerance > 0:
                color = tuple([
                    round(c / tolerance) * tolerance for c in color
                ])
            draw.line((x, 0, x, size[1]), fill=color)
    elif mode == 'vertical':
        for y in range(size[1]):
            pos = int(y * 100 / (size[1] - 1))
            color = spectrum[pos]
            if tolerance > 0:
                color = tuple([
                    round(c / tolerance) * tolerance for c in color
                ])
            draw.line((0, y, size[0], y), fill=color)

    blur = 1.5
    if size[0] > 512 or size[1] > 512:
        multiplier = max(size[0], size[1]) / 512
        if multiplier < 1.5:
            multiplier = 1.5
        blur = blur * multiplier

    img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return img
