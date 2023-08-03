import torch
from PIL import Image
from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
#model_path = "/mnt/pfs/users/yuzhipeng/shapeE/shap-e/01c65cec2a18462e9feef58b7fdfe627.obj"
#model_path = "/mnt/pfs/users/yuzhipeng/dataset/tank/middle_tank/35b9efb7c050408f92079d37aa3c8e55.glb"
model_path = "/mnt/pfs/users/yuzhipeng/dataset/tank/small_tank/86b0499b268349e09977b5dd0e006aea.glb"
model_path = "/mnt/pfs/users/yuzhipeng/dataset/Spaceship/big_tank/3144790491714609b88e951651099950.glb"
filename = model_path.split('/')[-1].split('.')[0]

# This may take a few minutes, since it requires rendering the model twice
# in two different modes.
batch = load_or_create_multimodal_batch(
    device,
    model_path=model_path,
    mv_light_mode="basic",
    mv_image_size=256,
    cache_dir="example_data/cactus/cached",
    verbose=True, # this will show Blender output during renders
)
with torch.no_grad():
    latent = xm.encoder.encode_to_bottleneck(batch)

    render_mode = 'stf' # you can change this to 'nerf'
    size = 512 # recommended that you lower resolution when using nerf

    cameras = create_pan_cameras(size, device)
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    
    for idx, image in enumerate(images):
        image.save(f'test/{idx}.png')    
    open(f'{filename}.html','w').write(gif_widget(images).value)
