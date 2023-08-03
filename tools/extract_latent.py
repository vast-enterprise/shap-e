import torch
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np
from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget


def extract_latent(xm, categery , model_path):
    filename = model_path.split('/')[-1].split('.')[0]
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
        latent_feat = latent.cpu().numpy()
        np.save(f'{categery}/{filename}.npz', latent_feat)
        render_mode = 'stf' # you can change this to 'nerf'
        size = 512 # recommended that you lower resolution when using nerf
        cameras = create_pan_cameras(size, device)
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        for idx, image in enumerate(images):
            image.save(f'test/{idx}.png')    
        open(f'{categery}/{filename}.html','w').write(gif_widget(images).value)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
categery = sys.argv[1]
datas = [line.strip() for line in open(sys.argv[2])]
for line in tqdm(datas):
    extract_latent(xm, categery, line)
