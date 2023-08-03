import torch
import sys
import numpy as np
from PIL import Image
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config, load_model_from_path
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image

VIEW_INDEXS = [1, 2, 3, 4]
VIEW_INDEXS = [6,6,6,6]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = sys.argv[1]
xm = load_model('transmitter', device=device)
model = load_model_from_path(
            'image300M', sys.argv[1],  device=device, config_path="/mnt/pfs/users/yuzhipeng/shapeE/shap-e/experiments/4view/image_cond_config.yaml")
diffusion = diffusion_from_config(load_config('diffusion'))
    
batch_size = 1
# guidance_scale = 15.0
guidance_scale = 3.0

# To get the best result, you should remove the background and show only the object of interest to the model.
# image = load_image("/mnt/pfs/data/zero_render/c6/c6ddfedc470f4667aeb8affceb6d2f6a/render_006.png")
images = []
for i in VIEW_INDEXS:
    image_path = f"/mnt/pfs/data/zero_render/c6/c6ddfedc470f4667aeb8affceb6d2f6a/render_{i:03d}.png"
    image = Image.open(image_path)
    img = image.convert('RGB')
    img = np.array(img)
    images.append(img)

images = np.stack(images, axis=0)
print(model,flush=True)


latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(images=[images]),
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
    images[0].save(f'{model_path}_{render_mode}_{i}.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)    
    # display(gif_widget(images))
    


# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh

for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'{model_path}_example_mesh_{render_mode}_{i}.ply', 'wb') as f:
        t.write_ply(f)
    with open(f'{model_path}_example_mesh_{render_mode}_{i}.obj', 'w') as f:
        t.write_obj(f)    
