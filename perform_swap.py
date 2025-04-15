import torch
import numpy as np
from typing import List, Dict, Optional
from PIL.Image import Image as PILImage
from IP_Adapter import IPAdapterXL

def compute_dataset_embeds_svd(
    all_embeds: np.ndarray,
    rank: int
) -> np.ndarray:
    # Perform SVD on the combined matrix
    _, _, v = np.linalg.svd(all_embeds, full_matrices=False)

    # Select the top `rank` singular vectors to construct the projection matrix
    v = v[:rank]
    projection_matrix = v.T @ v

    return projection_matrix

def get_projected_embedding(
    embed: np.ndarray,
    projection_matrix: np.ndarray
) -> np.ndarray:
    return embed @ projection_matrix

def get_embedding_composition(
    embed: np.ndarray,
    projections_data: List[Dict[str, np.ndarray]]
) -> np.ndarray:
    combined_embeds = embed.copy()

    for proj_data in projections_data:
        combined_embeds -= get_projected_embedding(embed, proj_data["projection_matrix"])
        combined_embeds += get_projected_embedding(proj_data["embed"], proj_data["projection_matrix"])

    return combined_embeds


def get_modified_images_embeds_composition(
    embed: np.ndarray,
    projections_data: List[Dict[str, np.ndarray]],
    ip_model: IPAdapterXL,
    prompt: Optional[str] = None,
    scale: float = 1.0,
    num_samples: int = 3,
    seed: int = 420,
    num_inference_steps: int = 50,
) -> List[PILImage]:
    final_embeds = get_embedding_composition(embed, projections_data)
    clip_embeds = torch.from_numpy(final_embeds)

    images: List[PILImage] = ip_model.generate(
    clip_image_embeds=clip_embeds,
    prompt=prompt,
    num_samples=num_samples,
    num_inference_steps=num_inference_steps,
    seed=seed,
    guidance_scale=7.5,
    scale=scale
    )
    return images




