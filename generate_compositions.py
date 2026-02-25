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
import gc
import json

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
    # phase: 'embed' to only compute & save CLIP embeddings and projection matrices
    # 'generate' to only run generation using saved embeddings; 'all' runs both
    phase: str = "all"

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


def _unload_model_from_gpu(model: Optional[torch.nn.Module]) -> None:
    """Try to move a model to CPU and free GPU memory.

    This is a best-effort helper: move model to CPU, delete reference,
    empty CUDA cache and run GC.
    """
    try:
        if model is not None:
            try:
                model.to('cpu')
            except Exception:
                # Some models/pipelines may not support .to('cpu'), ignore
                pass
            del model
    except Exception:
        pass
    # free up CUDA memory
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def process_combo(
    image_embeds_base_paths: List[str],
    image_names_base: List[str],
    concept_embeds_paths: List[List[str]],
    concept_names: List[List[str]],
    projection_matrices: List[np.ndarray],
    ip_model: IPAdapterXL,
    output_base_dir: str,
    num_samples: int,
    seed: int,
    prompt: Optional[str],
    scale: float
) -> None:
    for base_embed_path, base_name in zip(image_embeds_base_paths, image_names_base):
        # Load base embedding from .npy
        try:
            base_embed = np.load(base_embed_path)
        except Exception as e:
            print(f"Failed to load base embedding {base_embed_path}: {e}")
            continue

        # Generate all combinations of concept embeddings
        for combo_indices in np.ndindex(*(len(embeds) for embeds in concept_embeds_paths)):
            concept_combo_names = [concept_names[c][idx] for c, idx in enumerate(combo_indices)]
            combo_dir = os.path.join(
                output_base_dir,
                f"{base_name}_to_" + "_".join(concept_combo_names)
            )
            if os.path.exists(combo_dir):
                print(f"Directory {combo_dir} already exists. Skipping...")
                continue

            projections_data = []
            # For each selected concept image, load its saved embedding
            for c, idx in enumerate(combo_indices):
                embed_path = concept_embeds_paths[c][idx]
                try:
                    embed_arr = np.load(embed_path)
                except Exception as e:
                    print(f"Failed to load concept embedding {embed_path}: {e}")
                    embed_arr = None

                projections_data.append({
                    "embed": embed_arr,
                    "projection_matrix": projection_matrices[c]
                })

            modified_images = get_modified_images_embeds_composition(
                base_embed, projections_data, ip_model, prompt=prompt, scale=scale, num_samples=num_samples, seed=seed
            )
            save_images(combo_dir, modified_images)


def embed(concept_configs: List[ConceptConfig], cfg: MainConfig):
    # Stage 1: load CLIP, compute embeddings / projection matrices
    print("[embed] Loading CLIP model and computing embeddings...")
    model, preprocess = load_clip_model()

    image_files_base = [os.path.join(cfg.base_images_dir, f)
                        for f in os.listdir(cfg.base_images_dir)
                        if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Prepare directories for saving embeddings and projections
    embeddings_root = os.path.join(cfg.output_dir, "embeddings")
    base_emb_dir = os.path.join(embeddings_root, "base")
    concepts_emb_dir = os.path.join(embeddings_root, "concepts")
    projections_dir = os.path.join(embeddings_root, "projections")
    os.makedirs(base_emb_dir, exist_ok=True)
    os.makedirs(concepts_emb_dir, exist_ok=True)
    os.makedirs(projections_dir, exist_ok=True)

    image_embeds_base_paths = []
    image_names_base = []

    for path in image_files_base:
        img_name = os.path.basename(path)
        image_names_base.append(img_name)
        embed_arr = get_image_embeds(Image.open(path).convert("RGB"), model, preprocess)
        # save embedding to .npy
        save_path = os.path.join(base_emb_dir, f"{img_name}.npy")
        try:
            np.save(save_path, embed_arr)
        except Exception as e:
            print(f"Failed to save base embedding {save_path}: {e}")
        image_embeds_base_paths.append(save_path)

    concept_images_embeds_paths = []
    concept_images_names = []
    projection_matrices = []

    for concept in concept_configs:
        image_files = [os.path.join(concept.images_dir, f)
                       for f in os.listdir(concept.images_dir)
                       if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        embeds_paths = []
        names = []
        for path in image_files:
            img_name = os.path.basename(path)
            names.append(img_name)
            embed_arr = get_image_embeds(Image.open(path).convert("RGB"), model, preprocess)
            # save per-concept embedding
            concept_dir = os.path.join(concepts_emb_dir, concept.concept_name)
            os.makedirs(concept_dir, exist_ok=True)
            save_path = os.path.join(concept_dir, f"{img_name}.npy")
            try:
                np.save(save_path, embed_arr)
            except Exception as e:
                print(f"Failed to save concept embedding {save_path}: {e}")
            embeds_paths.append(save_path)
        concept_images_embeds_paths.append(embeds_paths)
        concept_images_names.append(names)

        with open(concept.embeddings_path, "rb") as f:
            all_embeds_in = np.load(f)
        projection_matrix = compute_dataset_embeds_svd(all_embeds_in, concept.rank)
        projection_matrices.append(projection_matrix)
        # save projection matrix so generation can be run separately later
        proj_save_path = os.path.join(projections_dir, f"{concept.concept_name}.npy")
        try:
            np.save(proj_save_path, projection_matrix)
        except Exception as e:
            print(f"Failed to save projection matrix {proj_save_path}: {e}")

    # Unload CLIP model and free GPU memory so next heavy model can be loaded
    print("[embed] Finished embeddings; unloading CLIP model from GPU...")
    _unload_model_from_gpu(model)

    # Save metadata describing embedding file paths and names
    metadata = {
        "base_emb_paths": image_embeds_base_paths,
        "base_names": image_names_base,
        "concept_emb_paths": concept_images_embeds_paths,
        "concept_names": concept_images_names,
        "projection_files": [os.path.join(projections_dir, f"{c.concept_name}.npy") for c in concept_configs]
    }
    metadata_path = os.path.join(embeddings_root, "metadata.json")
    try:
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2)
    except Exception as e:
        print(f"Failed to write metadata {metadata_path}: {e}")

    return image_embeds_base_paths, image_names_base, concept_images_embeds_paths, concept_images_names, projection_matrices


def load_saved_embeddings(cfg: MainConfig):
    """Load embedding paths, names, and projection matrices from disk (metadata)."""
    embeddings_root = os.path.join(cfg.output_dir, "embeddings")
    metadata_path = os.path.join(embeddings_root, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run embed phase first.")
    with open(metadata_path, "r", encoding="utf-8") as mf:
        metadata = json.load(mf)

    base_emb_paths = metadata.get("base_emb_paths", [])
    base_names = metadata.get("base_names", [])
    concept_emb_paths = metadata.get("concept_emb_paths", [])
    concept_names = metadata.get("concept_names", [])
    projection_files = metadata.get("projection_files", [])

    projection_matrices = []
    for pf in projection_files:
        if not os.path.exists(pf):
            raise FileNotFoundError(f"Projection file not found: {pf}")
        projection_matrices.append(np.load(pf))

    return base_emb_paths, base_names, concept_emb_paths, concept_names, projection_matrices

def main(cfg: MainConfig):
    concept_configs = [ConceptConfig(**c) for c in cfg.concepts]
    # Support three modes: 'embed' (compute & save embeddings),
    # 'generate' (load saved embeddings and run generation), or 'all'.
    if cfg.phase in ("embed", "all"):
        image_embeds_base_paths, image_names_base, concept_images_embeds_paths, concept_images_names, projection_matrices = embed(concept_configs, cfg=cfg)
    else:
        # when only generating, load previously saved metadata & projections
        image_embeds_base_paths, image_names_base, concept_images_embeds_paths, concept_images_names, projection_matrices = load_saved_embeddings(cfg)

    if cfg.phase in ("generate", "all"):
        ip_model = load_ip_adapter_model()

        process_combo(
            image_embeds_base_paths,
            image_names_base,
            concept_images_embeds_paths,
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