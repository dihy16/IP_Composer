import os
import itertools
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage
from PIL.ImageFont import FreeTypeFont
from PIL.ImageDraw import ImageDraw as PILImageDraw

def load_image(path: str) -> PILImage:
            return Image.open(path) if os.path.exists(path) else Image.new("RGB", (256, 256), color="white")


def wrap_text(text: str, max_width: int, draw: PILImageDraw, font: FreeTypeFont) -> List[str]:
    """
    Wrap the text to fit within the given width by breaking it into lines.
    """
    lines = []
    words = text.split(' ')
    current_line = []

    for word in words:
        current_line.append(word)
        line_width = draw.textbbox((0, 0), ' '.join(current_line), font=font)[2]
        if line_width > max_width:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines

def image_grid_with_titles(
    imgs: List[PILImage],
    rows: int,
    cols: int,
    top_titles: List[str],
    left_titles: List[str],
    margin: int = 20
) -> PILImage:
    assert len(imgs) == rows * cols
    assert len(top_titles) == cols
    assert len(left_titles) == rows

    imgs = [img.resize((256, 256)) for img in imgs]
    w, h = imgs[0].size

    title_height = 50
    title_width = 120

    grid_width = cols * (w + margin) + title_width + margin
    grid_height = rows * (h + margin) + title_height + margin

    grid = Image.new('RGB', size=(grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for i, title in enumerate(top_titles):
        wrapped_title = wrap_text(title, w, draw, font)
        total_text_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in wrapped_title])
        y_offset = (title_height - total_text_height) // 2

        for line in wrapped_title:
            text_width = draw.textbbox((0, 0), line, font=font)[2]
            x_offset = ((i * (w + margin)) + title_width + margin + (w - text_width) // 2)
            draw.text((x_offset, y_offset), line, fill="black", font=font)
            y_offset += draw.textbbox((0, 0), line, font=font)[3]

    for i, title in enumerate(left_titles):
        wrapped_title = wrap_text(title, title_width - 10, draw, font)
        total_text_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in wrapped_title])
        y_offset = (i * (h + margin)) + title_height + (h - total_text_height) // 2 + margin

        for line in wrapped_title:
            text_width = draw.textbbox((0, 0), line, font=font)[2]
            x_offset = (title_width - text_width) // 2
            draw.text((x_offset, y_offset), line, fill="black", font=font)
            y_offset += draw.textbbox((0, 0), line, font=font)[3]

    for i, img in enumerate(imgs):
        x_pos = (i % cols) * (w + margin) + title_width + margin
        y_pos = (i // cols) * (h + margin) + title_height + margin
        grid.paste(img, box=(x_pos, y_pos))

    return grid


def assemble_and_save_grid(
    images: List[PILImage],
    left_titles: List[str],
    top_titles: List[str],
    output_path: str
) -> None:
    total_required = len(left_titles) * len(top_titles)
    if len(images) < total_required:
        images.extend([Image.new("RGB", (256, 256), color="white")] * (total_required - len(images)))

    grid = image_grid_with_titles(
        imgs=images,
        rows=len(left_titles),
        cols=len(top_titles),
        top_titles=top_titles,
        left_titles=left_titles
    )

    grid.save(output_path)

def create_grids(
    base_images_dir: str,
    concept_images_dirs: List[str],
    concept_names: List[str],
    output_dir: str,
    num_samples: int
) -> None:
    output_grid_dir = os.path.join(output_dir, "grids")
    os.makedirs(output_grid_dir, exist_ok=True)

    base_images = os.listdir(base_images_dir)

    fixed_concept_dirs = concept_images_dirs[:-1]
    final_concept_dir = concept_images_dirs[-1]
    final_concept_images = os.listdir(final_concept_dir)

    top_titles = ["Base Image"] + concept_names + ["Samples"] + [""] * (num_samples - 1)
    left_titles = [""] * len(final_concept_images)

    fixed_concept_images_list = [os.listdir(d) for d in fixed_concept_dirs]
    fixed_combinations = itertools.product(*fixed_concept_images_list) if fixed_concept_images_list else [()]

    for base_image in base_images:
        base_image_path = os.path.join(base_images_dir, base_image)

        for fixed_combination in fixed_combinations:
            images = []
            fixed_images = [load_image(base_image_path)]

            for concept_dir, concept_image in zip(fixed_concept_dirs, fixed_combination):
                concept_image_path = os.path.join(concept_dir, concept_image)
                fixed_images.append(load_image(concept_image_path))

            for final_image in final_concept_images:
                final_image_path = os.path.join(final_concept_dir, final_image)
                row_images = fixed_images + [load_image(final_image_path)]

                # Sample directory is constructed from concept order
                parts = [base_image] + list(fixed_combination) + [final_image]
                sample_dir_name = "_to_".join([parts[0], "_".join(parts[1:])])
                sample_dir = os.path.join(output_dir, sample_dir_name)

                if os.path.exists(sample_dir):
                    sample_images = sorted(os.listdir(sample_dir))
                    row_images.extend([load_image(os.path.join(sample_dir, img)) for img in sample_images])

                images.extend(row_images)

            combo_name = "_".join(fixed_combination)
            if combo_name:
                grid_filename = f"{base_image}_{combo_name}_grid.png"
            else:
                grid_filename = f"{base_image}_grid.png"

            grid_save_path = os.path.join(output_grid_dir, grid_filename)
            
            assemble_and_save_grid(images, left_titles, top_titles, grid_save_path)


