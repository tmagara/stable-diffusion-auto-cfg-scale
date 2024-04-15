import pathlib
import torch
import diffusers

from . import pipeline


results_folder =  pathlib.Path('result')
results_folder.mkdir(parents=True, exist_ok=True)

pipe = pipeline.CustomStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16",
    vae=diffusers.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
)
pipe.scheduler = diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.enable_model_cpu_offload()

# width, height = 1024, 1024
width, height = 1344, 768

# prompt = ["portrait, natural light, ultra detailed"]
# prompt = ['a girl at a botanical cafe, fancy translucent uniform, necktie, bare shoulders, actual location, layered composition, atmospheric perspective']
prompt = ["cute animals in victorian attire, village life, feltwork, handcraft, historical authenticity"]
# prompt = ['a girl at an abandoned station, vivid eyes, realistic face, hair ornament, alice blue attire, water flooded, natural light']
negative_prompt = ['unknown author, random sketch']

generator = torch.Generator("cuda")
seed = generator.seed()
print(f"seed = {seed}")
# seed=3086457027005092

for label, num_inference_steps, kwargs in [
    ("normalize", 15, { 'guidance_scale': 1.125, 'cfg_normalize': True, 'boost_scale': 0.125}),
    # ("normalize", 15, { 'guidance_scale': 1.25, 'cfg_normalize': True, 'boost_scale': 0.0625}),
]:
    generator = generator.manual_seed(seed)
    for i in range(100):
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
