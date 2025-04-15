import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import open_clip
from huggingface_hub import hf_hub_download
from IP_Adapter import IPAdapterXL
from perform_swap import compute_dataset_embeds_svd, get_modified_images_embeds_composition
from create_grids import create_grids
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pyrallis
from torchvision.transforms import Compose
from PIL.Image import Image as PILImage

@dataclass
class ConceptConfig:
    concept_name: str
    images_dir: str
    embeddings_path: str
    rank: int

@dataclass
class MainConfig:
    base_images_dir: str
    concepts: List[dict]
    output_dir: str
    seed: int = 420
    prompt: Optional[str] = None
    scale: float = 1.0
    num_samples: int = 4
    create_grids: bool = False

def save_images(output_dir: str, image_list: List[PILImage]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(image_list):
        img.save(os.path.join(output_dir, f"sample_{i + 1}.png"))


def get_image_embeds(
    pil_image: PILImage,
    model: torch.nn.Module,
    preprocess: Compose,
    device: str = "cuda"
) -> np.ndarray:
    image = preprocess(pil_image)[np.newaxis, :, :, :]
    with torch.no_grad():
        embeds = model.encode_image(image.to(device))
    return embeds.cpu().detach().numpy()

def load_ip_adapter_model() -> IPAdapterXL:
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )

    image_encoder_repo = 'h94/IP-Adapter'
    image_encoder_subfolder = 'models/image_encoder'
    ip_ckpt = hf_hub_download('h94/IP-Adapter', subfolder="sdxl_models", filename='ip-adapter_sdxl_vit-h.bin')
    
    device = "cuda"
    ip_model = IPAdapterXL(pipe, image_encoder_repo, image_encoder_subfolder, ip_ckpt, device)

    return ip_model

def load_clip_model() -> Tuple[torch.nn.Module, Compose]:
    device = 'cuda:0'
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    model.to(device)
    return model, preprocess

def process_combo(
    image_embeds_base: List[np.ndarray],
    image_names_base: List[str],
    concept_embeds: List[List[np.ndarray]],
    concept_names: List[List[str]],
    projection_matrices: List[np.ndarray],
    ip_model: IPAdapterXL,
    output_base_dir: str,
    num_samples: int,
    seed: int,
    prompt: Optional[str],
    scale: float
) -> None:
    for base_embed, base_name in zip(image_embeds_base, image_names_base):
        # Generate all combinations of concept embeddings
        for combo_indices in np.ndindex(*(len(embeds) for embeds in concept_embeds)):
            concept_combo_names = [concept_names[c][idx] for c, idx in enumerate(combo_indices)]
            combo_dir = os.path.join(
                output_base_dir,
                f"{base_name}_to_" + "_".join(concept_combo_names)
            )
            if os.path.exists(combo_dir):
                print(f"Directory {combo_dir} already exists. Skipping...")
                continue

            projections_data = [
                {
                    "embed": concept_embeds[c][idx],
                    "projection_matrix": projection_matrices[c]
                }
                for c, idx in enumerate(combo_indices)
            ]

            modified_images = get_modified_images_embeds_composition(
                base_embed, projections_data, ip_model, prompt=prompt, scale=scale, num_samples=num_samples, seed=seed
            )
            save_images(combo_dir, modified_images)


def main(cfg: MainConfig):
    concept_configs = [ConceptConfig(**c) for c in cfg.concepts]

    ip_model = load_ip_adapter_model()
    model, preprocess = load_clip_model()

    image_files_base = [os.path.join(cfg.base_images_dir, f)
                        for f in os.listdir(cfg.base_images_dir)
                        if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    image_embeds_base = []
    image_names_base = []

    for path in image_files_base:
        img_name = os.path.basename(path)
        image_names_base.append(img_name)
        image_embeds_base.append(get_image_embeds(Image.open(path).convert("RGB"), model, preprocess))

    concept_images_embeds = []
    concept_images_names = []
    projection_matrices = []

    for concept in concept_configs:
        image_files = [os.path.join(concept.images_dir, f)
                       for f in os.listdir(concept.images_dir)
                       if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        embeds = []
        names = []
        for path in image_files:
            img_name = os.path.basename(path)
            names.append(img_name)
            embeds.append(get_image_embeds(Image.open(path).convert("RGB"), model, preprocess))
        concept_images_embeds.append(embeds)
        concept_images_names.append(names)

        with open(concept.embeddings_path, "rb") as f:
            all_embeds_in = np.load(f)
        projection_matrix = compute_dataset_embeds_svd(all_embeds_in, concept.rank)
        projection_matrices.append(projection_matrix)

    process_combo(
        image_embeds_base,
        image_names_base,
        concept_images_embeds,
        concept_images_names,
        projection_matrices,
        ip_model,
        cfg.output_dir,
        cfg.num_samples,
        cfg.seed,
        cfg.prompt,
        cfg.scale
    )

    if cfg.create_grids:
        concept_dirs = [c.images_dir for c in concept_configs]
        concept_names = [c.concept_name for c in concept_configs]
        create_grids(cfg.base_images_dir, concept_dirs, concept_names, cfg.output_dir, cfg.num_samples)


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=MainConfig)
    main(cfg)