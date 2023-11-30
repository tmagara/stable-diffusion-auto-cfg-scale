import pathlib
import torch
import diffusers

from . import pipeline


save_path = '/tmp/volatile/'
results_folder = pathlib.Path(save_path)
results_folder.mkdir(exist_ok=True)

pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
)

pipe = pipeline.CustomStableDiffusionXLPipeline(
    pipe.vae,
    pipe.text_encoder,
    pipe.text_encoder_2,
    pipe.tokenizer,
    pipe.tokenizer_2,
    pipe.unet,
    pipe.scheduler,
).to("cuda")

num_inference_steps = 10
width, height, prompt, negative_prompt = (
    1024, 1024, 

    # short simple prompt
    ["portrait, natural light, ultra detailed"], 
    ["cartoon, mature, abuse, funny limbs, 3D CGI, blurry, faded, monochrome, watermark"],

    # ice queen prompt from <https://openaijourney.com/best-sdxl-models/>
    # ["(fractal cystal skin:1.1) with( ice crown:1.4) woman, white crystal skin, (fantasy:1.3), (Anna Dittmann:1.3)"],
    # ["blurry, blur, text, watermark, painting, anime, cartoon, render, 3d, nsfw, nude"],
)

generator = torch.Generator("cuda")
seed = generator.seed()
# seed = 42
print(f"seed = {seed}")

for label, kwargs in [
    ("auto", {'auto_guidance_scale': True}), 
    ("cfg4", {'auto_guidance_scale': False, 'guidance_scale': 4.0}), 
    ("cfg7", {'auto_guidance_scale': False, 'guidance_scale': 7.0}),
    ("cfg15", {'auto_guidance_scale': False, 'guidance_scale': 15.0}),
]:
    generator = generator.manual_seed(seed)
    for i in range(10):
        image = pipe(
            prompt=prompt, 
            height=height, 
            width=width, 
            num_inference_steps=num_inference_steps, 
            negative_prompt=negative_prompt, 
            generator=generator,
            **kwargs
        ).images[0]
        image.save(results_folder / f'{label}_{i}.png')
