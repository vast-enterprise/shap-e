import sys
import torch
from PIL import Image
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
#from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.models.download import load_model, load_config, load_model_from_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
if len(sys.argv) <2:
    model_path = 'shape_base'
    model = load_model('text300M', device=device)
else:
    model_path = sys.argv[1]
    model = load_model_from_path('text300M', sys.argv[1], device=device)

#model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
batch_size = 4
guidance_scale = 15.0
prompt = "Medium-sized warship equipped with four Gauss cannons"

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

render_mode = 'nerf' # you can change this to 'stf'
size = 64 # this is the size of the renders; higher values take longer to render.

cameras = create_pan_cameras(size, device)
for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    # Image.save(f'{i}.gif', save_all=True, append_images=images)
    images[0].save(f'{model_path}_{i}.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)    
    # display(gif_widget(images))
    


# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh

for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'{model_path}_example_mesh_{i}.ply', 'wb') as f:
        t.write_ply(f)
    with open(f'{model_path}_example_mesh_{i}.obj', 'w') as f:
        t.write_obj(f)    
