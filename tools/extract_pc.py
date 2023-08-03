import torch
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np
from shap_e.models.download import load_model
import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional, Union
from shap_e.rendering.point_cloud import PointCloud
from .data_util import load_or_create_pc


def extract_latent(xm, categery , model_path):
    filename = model_path.split('/')[-1].split('.')[0]
    cache_dir="example_data/cactus/cached",
    pc_num_views=40
    point_count=2**14
    random_sample_count = 2**19,
    pc = load_or_create_pc(
        # mesh_path=mesh_path,
        model_path=model_path,
        cache_dir=cache_dir,
        random_sample_count=random_sample_count,
        point_count=point_count,
        num_views=pc_num_views,
        verbose=True,
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
categery = sys.argv[1]
datas = [line.strip() for line in open(sys.argv[2])]
for line in tqdm(datas):
    extract_latent(xm, categery, line)
