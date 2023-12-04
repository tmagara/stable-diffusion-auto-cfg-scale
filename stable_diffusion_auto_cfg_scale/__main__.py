import pathlib
import torch
import diffusers

from . import pipeline


results_folder =  pathlib.Path('result')

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

# pipe.scheduler = diffusers.schedulers.DDIMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear", timestep_spacing="trailing", set_alpha_to_one=True)

width, height = 1024, 1024

# short simple prompt
prompt = ["portrait, natural light, ultra detailed"]
negative_prompt = ["cartoon, mature, abuse, blurry, faded, monochrome, watermark"]

# # Retro Photography prompt from <https://openaijourney.com/best-sdxl-prompts/>
# prompt = ["retro style, 90s photo of a captivating girl having lunch in a restaurant, a bustling metropolis, neon barrettes, enigmatic setting, retro"]
# negative_prompt = ["blurry, blur, text, watermark, render, 3D, NSFW, nude, CGI, monochrome, B&W, cartoon, painting, smooth, plastic, blurry, low-resolution, deep-fried, oversaturated"]

# # Ice Queen prompt from <https://openaijourney.com/best-sdxl-prompts/>
# prompt = ["(fractal crystal skin:1.1) with( ice crown:1.4) woman, white crystal skin, (fantasy:1.3), (Anna Dittmann:1.3)"]
# negative_prompt = ["blurry, blur, text, watermark, painting, anime, cartoon, render, 3d, nsfw, nude"]

# # Rabbit 3D Render   prompt from <https://openaijourney.com/best-sdxl-prompts/>
# prompt = ["Cute rabbit wearing a jacket, eating a carrot, 3D Style, rendering"]
# negative_prompt = ["blurry, blur, text, watermark, render, 3D, NSFW, nude, CGI, monochrome, B&W, cartoon, painting, smooth, plastic, blurry, low-resolution, deep-fried, oversaturated"]

generator = torch.Generator("cuda")
seed = generator.seed()
# seed = 42
print(f"seed = {seed}")

for label, num_inference_steps, kwargs in [
    ("auto_30steps", 30, {'auto_guidance_scale': True}), 
    ("cfg1_30steps", 30, {'auto_guidance_scale': False, 'guidance_scale': 1.0}), 
    ("cfg2_30steps", 30, {'auto_guidance_scale': False, 'guidance_scale': 2.0}), 
    ("cfg5_30steps", 30, {'auto_guidance_scale': False, 'guidance_scale': 5.0}),
    ("cfg15_30steps", 30, {'auto_guidance_scale': False, 'guidance_scale': 15.0}),
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
