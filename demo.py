import torch
import numpy as np
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import open_clip
from huggingface_hub import hf_hub_download
from IP_Adapter.ip_adapter import IPAdapterXL
from perform_swap import compute_dataset_embeds_svd, get_modified_images_embeds_composition
from generate_text_embeddings import load_descriptions, generate_embeddings
import random
import spaces
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize SDXL pipeline
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)

# Initialize IP-Adapter
image_encoder_repo = 'h94/IP-Adapter'
image_encoder_subfolder = 'models/image_encoder'
ip_ckpt = hf_hub_download('h94/IP-Adapter', subfolder="sdxl_models", filename='ip-adapter_sdxl_vit-h.bin')
ip_model = IPAdapterXL(pipe, image_encoder_repo, image_encoder_subfolder, ip_ckpt, device)

# Initialize CLIP model
clip_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
clip_model.to(device)
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

CONCEPTS_MAP = {
    "age": "age_descriptions.npy",
    "animal fur": "fur_descriptions.npy",
    "dogs": "dog_descriptions.npy",
    "emotions": "emotion_descriptions.npy",
    "flowers": "flower_descriptions.npy",
    "fruit/vegtable": "fruit_vegetable_descriptions.npy",
    "outfit type": "outfit_descriptions.npy",
    "outfit pattern (including color)": "outfit_pattern_descriptions.npy",
    "patterns": "pattern_descriptions.npy",
    "patterns (including color)": "pattern_descriptions_with_colors.npy",
    "vehicle": "vehicle_descriptions.npy",
    "daytime": "times_of_day_descriptions.npy",
    "pose": "person_poses_descriptions.npy",
    "season": "season_descriptions.npy",
    "material": "material_descriptions.npy"
}
RANKS_MAP = {
    "age": 30,
    "animal fur": 80,
    "dogs": 30,
    "emotions": 30,
    "flowers": 30,
    "fruit/vegtable": 30,
    "outfit type": 30,
    "outfit pattern (including color)": 80,
    "patterns": 80,
    "patterns (including color)": 80,
    "vehicle": 30,
    "daytime": 30,
    "pose": 30,
    "season": 30,
    "material": 80,
}
concept_options = list(CONCEPTS_MAP.keys())


MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def change_rank_default(concept_name):
    return RANKS_MAP.get(concept_name, 30)


@spaces.GPU
def get_image_embeds(pil_image, model=clip_model, preproc=preprocess, dev=device):
    """Get CLIP image embeddings for a given PIL image"""
    image = preproc(pil_image)[np.newaxis, :, :, :]
    with torch.no_grad():
        embeds = model.encode_image(image.to(dev))
    return embeds.cpu().detach().numpy()


def process_images(
        base_image,
        concept_image1, concept_name1,
        concept_image2=None, concept_name2=None,
        concept_image3=None, concept_name3=None,
        rank1=10, rank2=10, rank3=10,
        prompt=None,
        scale=1.0,
        seed=420,
        num_inference_steps=50,
        concpet_from_file_1=None,
        concpet_from_file_2=None,
        concpet_from_file_3=None,
        use_concpet_from_file_1=False,
        use_concpet_from_file_2=False,
        use_concpet_from_file_3=False
):
    """Process the base image and concept images to generate modified images"""
    # Process base image
    base_image_pil = Image.fromarray(base_image).convert("RGB")
    base_embed = get_image_embeds(base_image_pil, clip_model, preprocess, device)

    # Process concept images
    concept_images = []
    concept_descriptions = []

    skip_load_concept = [False, False, False]

    # for demo purposes we allow for up to 3 different concepts and corresponding concept images
    if concept_image1 is not None:
        concept_images.append(concept_image1)
        if use_concpet_from_file_1 and concpet_from_file_1 is not None:  # if concept is new from user input
            concept_descriptions.append(concpet_from_file_1)
            skip_load_concept[0] = True
        else:
            concept_descriptions.append(CONCEPTS_MAP[concept_name1])
    else:
        return None, "Please upload at least one concept image"

    # Add second concept (optional)
    if concept_image2 is not None:
        concept_images.append(concept_image2)
        if use_concpet_from_file_2 and concpet_from_file_2 is not None:  # if concept is new from user input
            concept_descriptions.append(concpet_from_file_2)
            skip_load_concept[1] = True
        else:
            concept_descriptions.append(CONCEPTS_MAP[concept_name2])

    # Add third concept (optional)
    if concept_image3 is not None:
        concept_images.append(concept_image3)
        if use_concpet_from_file_3 and concpet_from_file_3 is not None:  # if concept is new from user input
            concept_descriptions.append(concpet_from_file_3)
            skip_load_concept[2] = True
        else:
            concept_descriptions.append(CONCEPTS_MAP[concept_name3])

    # Get all ranks
    ranks = [rank1]
    if concept_image2 is not None:
        ranks.append(rank2)
    if concept_image3 is not None:
        ranks.append(rank3)

    concept_embeds = []
    projection_matrices = []
    # for the demo, we assume 1 concept image per concept
    # for each concept image, we calculate it's image embeedings and load the concepts textual embeddings to copmpute the projection matrix over it
    for i, concept in enumerate(concept_descriptions):
        img_pil = Image.fromarray(concept_images[i]).convert("RGB")
        concept_embeds.append(get_image_embeds(img_pil, clip_model, preprocess, device))
        if skip_load_concept[i]:  # if concept is new from user input
            all_embeds_in = concept
        else:
            embeds_path = f"./text_embeddings/{concept}"
            with open(embeds_path, "rb") as f:
                all_embeds_in = np.load(f)

        projection_matrix = compute_dataset_embeds_svd(all_embeds_in, ranks[i])
        projection_matrices.append(projection_matrix)

    # Create projection data structure for the composition
    projections_data = [
        {
            "embed": embed,
            "projection_matrix": proj_matrix
        }
        for embed, proj_matrix in zip(concept_embeds, projection_matrices)
    ]

    # Generate modified images -
    modified_images = get_modified_images_embeds_composition(
        base_embed,
        projections_data,
        ip_model,
        prompt=prompt,
        scale=scale,
        num_samples=1,
        seed=seed,
        num_inference_steps=num_inference_steps
    )

    return modified_images[0]


def get_text_embeddings(concept_file):
    print("generating text embeddings")
    descriptions = load_descriptions(concept_file)
    embeddings = generate_embeddings(descriptions, clip_model, tokenizer, device, batch_size=100)
    print("text embeddings shape", embeddings.shape)
    return embeddings, True


def process_and_display(
        base_image,
        concept_image1, concept_name1="age",
        concept_image2=None, concept_name2=None,
        concept_image3=None, concept_name3=None,
        rank1=30, rank2=30, rank3=30,
        prompt=None, scale=1.0, seed=0, num_inference_steps=50,
        concpet_from_file_1=None,
        concpet_from_file_2=None,
        concpet_from_file_3=None,
        use_concpet_from_file_1=False,
        use_concpet_from_file_2=False,
        use_concpet_from_file_3=False
):
    if base_image is None:
        raise gr.Error("Please upload a base image")

    if concept_image1 is None:
        raise gr.Error("Choose at least one concept image")

    if concept_image1 is None:
        raise gr.Error("Choose at least one concept type")

    modified_images = process_images(
        base_image,
        concept_image1, concept_name1,
        concept_image2, concept_name2,
        concept_image3, concept_name3,
        rank1, rank2, rank3,
        prompt, scale, seed, num_inference_steps,
        concpet_from_file_1,
        concpet_from_file_2,
        concpet_from_file_3,
        use_concpet_from_file_1,
        use_concpet_from_file_2,
        use_concpet_from_file_3
    )

    return modified_images


# UI CSS
css = """
#col-container {
    margin: 0 auto;
    max-width: 800px;
}
"""
example = """
Emotion Description

a photo of a person feeling joyful

a photo of a person feeling sorrowful

a photo of a person feeling enraged

a photo of a person feeling astonished

a photo of a person feeling disgusted

a photo of a person feeling terrified

...

"""
with gr.Blocks(css=css) as demo:
    gr.Markdown(f"""# IP Composer 🌅✚🖌️
### compose new images with visual concepts extracted from refrence images using CLIP & IP Adapter


#### 🛠️ How to Use:                                   
1. Upload a base image  
2. Upload 1–3 concept images   
3. Select a **concept type** to extract from each concept image:  
    - Choose a **predefined concept type** from the dropdown (e.g. pattern, emotion, pose), **or**  
    - Upload a **file with text variations of your concept** (e.g. prompts from an LLM).  
        - 👉 If you're uploading a **new concept**, don't forget to **adjust the "rank" value** under **Advanced Options** for better results.

Following the algorithm proposed in IP-Composer: Semantic Composition of Visual Concepts by Dorfman et al.
[[Project page](https://ip-composer.github.io/IP-Composer/)] [[arxiv](https://arxiv.org/pdf/2502.13951)]
        """)
    concpet_from_file_1 = gr.State()
    concpet_from_file_2 = gr.State()
    concpet_from_file_3 = gr.State()
    use_concpet_from_file_1 = gr.State()
    use_concpet_from_file_2 = gr.State()
    use_concpet_from_file_3 = gr.State()
    with gr.Row():
        with gr.Column():
            base_image = gr.Image(label="Base Image (Required)", type="numpy")
            with gr.Tab("Concept 1"):
                with gr.Group():
                    concept_image1 = gr.Image(label="Concept Image 1", type="numpy")
                    with gr.Row():
                        concept_name1 = gr.Dropdown(concept_options, label="Concept 1", value=None,
                                                    info="Pick concept type")
                        with gr.Accordion("💡 Or use a new concept 👇", open=False):
                            gr.Markdown("1. Upload a file with text variations of your concept (e.g. ask an LLM)")
                            gr.Markdown("2. Prefereably with > 100 variations.")
                            with gr.Accordion("File example for the concept 'emotions'", open=False):
                                gr.Markdown(example)
                            concept_file_1 = gr.File(label="Concept variations", file_types=["text"])

            with gr.Tab("Concept 2 (Optional)"):
                with gr.Group():
                    concept_image2 = gr.Image(label="Concept Image 2", type="numpy")
                    with gr.Row():
                        concept_name2 = gr.Dropdown(concept_options, label="Concept 2", value=None,
                                                    info="Pick concept type")
                        with gr.Accordion("💡 Or use a new concept 👇", open=False):
                            gr.Markdown("1. Upload a file with text variations of your concept (e.g. ask an LLM)")
                            gr.Markdown("2. Prefereably with > 100 variations.")
                            with gr.Accordion("File example for the concept 'emotions'", open=False):
                                gr.Markdown(example)
                            concept_file_2 = gr.File(label="Concept variations", file_types=["text"])

            with gr.Tab("Concept 3 (optional)"):
                with gr.Group():
                    concept_image3 = gr.Image(label="Concept Image 3", type="numpy")
                    with gr.Row():
                        concept_name3 = gr.Dropdown(concept_options, label="Concept 3", value=None,
                                                    info="Pick concept type")
                        with gr.Accordion("💡 Or use a new concept 👇", open=False):
                            gr.Markdown("1. Upload a file with text variations of your concept (e.g. ask an LLM)")
                            gr.Markdown("2. Prefereably with > 100 variations.")
                            with gr.Accordion("File example for the concept 'emotions'", open=False):
                                gr.Markdown(example)
                            concept_file_3 = gr.File(label="Concept variations", file_types=["text"])

            with gr.Accordion("Advanced options", open=False):
                prompt = gr.Textbox(label="Guidance Prompt (Optional)",
                                    placeholder="Optional text prompt to guide generation")
                num_inference_steps = gr.Slider(minimum=1, maximum=50, value=30, step=1, label="Num steps")
                with gr.Row():
                    scale = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Scale")
                    randomize_seed = gr.Checkbox(value=True, label="Randomize seed")
                    seed = gr.Number(value=0, label="Seed", precision=0)
                with gr.Column():
                    gr.Markdown("If a concept is not showing enough, try to increase the rank")
                    with gr.Row():
                        rank1 = gr.Slider(minimum=1, maximum=150, value=30, step=1, label="Rank concept 1")
                        rank2 = gr.Slider(minimum=1, maximum=150, value=30, step=1, label="Rank concept 2")
                        rank3 = gr.Slider(minimum=1, maximum=150, value=30, step=1, label="Rank concept 3")

        with gr.Column():
            output_image = gr.Image(label="Composed output", show_label=True)
            submit_btn = gr.Button("Generate")


    concept_file_1.upload(
        fn=get_text_embeddings,
        inputs=[concept_file_1],
        outputs=[concpet_from_file_1, use_concpet_from_file_1]
    )
    concept_file_2.upload(
        fn=get_text_embeddings,
        inputs=[concept_file_2],
        outputs=[concpet_from_file_2, use_concpet_from_file_2]
    )
    concept_file_3.upload(
        fn=get_text_embeddings,
        inputs=[concept_file_3],
        outputs=[concpet_from_file_3, use_concpet_from_file_3]
    )

    concept_file_1.delete(
        fn=lambda x: False,
        inputs=[concept_file_1],
        outputs=[use_concpet_from_file_1]
    )
    concept_file_2.delete(
        fn=lambda x: False,
        inputs=[concept_file_2],
        outputs=[use_concpet_from_file_2]
    )
    concept_file_3.delete(
        fn=lambda x: False,
        inputs=[concept_file_3],
        outputs=[use_concpet_from_file_3]
    )

    submit_btn.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
    ).then(fn=process_and_display,
           inputs=[
               base_image,
               concept_image1, concept_name1,
               concept_image2, concept_name2,
               concept_image3, concept_name3,
               rank1, rank2, rank3,
               prompt, scale, seed, num_inference_steps,
               concpet_from_file_1,
               concpet_from_file_2,
               concpet_from_file_3,
               use_concpet_from_file_1,
               use_concpet_from_file_2,
               use_concpet_from_file_3
           ],
           outputs=[output_image]
           )

    concept_name1.select(
        fn=change_rank_default,
        inputs=[concept_name1],
        outputs=[rank1]
    )
    concept_name2.select(
        fn=change_rank_default,
        inputs=[concept_name2],
        outputs=[rank2]
    )
    concept_name3.select(
        fn=change_rank_default,
        inputs=[concept_name3],
        outputs=[rank3]
    )

demo.launch(share=True)