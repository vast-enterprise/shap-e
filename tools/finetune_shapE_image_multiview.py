# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: June 22, 2023
#
# This code is licensed under the MIT License.
#
# If you use or modify this code in your project, we kindly ask that you include
# this copyright notice in each file where the code is used. Your cooperation helps
# acknowledge the effort that went into creating this project.
# ==============================================================================


import torch
import torch.optim as optim

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config, load_config_frompath, load_model_from_path
from shap_e.models.configs import model_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from IPython import embed

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import argparse

import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import csv
import time
import random
import numpy as np
from datetime import datetime
from PIL import Image


def setup_ddp(gpu, args):
    dist.init_process_group(
        backend='nccl',      # backend='gloo',#
        init_method='env://',
        world_size=args.world_size,
        rank=gpu)

    torch.cuda.set_device(gpu)


RENDER_PATH = '/mnt/pfs/data/zero_render/'
VIEW_INDEXS = [1, 2, 3, 4]

class Mydataset(Dataset):
    def __init__(self, latent_code_path):
        self.captions = [line.strip().split(',') for line in open(
            '/mnt/pfs/users/yuzhipeng/shapeE/shap-e/dataset/train.txt')]
        self.latent_code_path = latent_code_path

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        uuid, caption = self.captions[i]
        images = []
        if args.random_view:
            view_indexes = [random.randint(1, 24) for _ in range(4)]
        else:
            view_indexes = VIEW_INDEXS
           
        for image_index in view_indexes:
            # image_index = random.randint(1,24)
            image_path = f'{RENDER_PATH}/{uuid[:2]}/{uuid}/render_{image_index:03d}.png'
            image = Image.open(image_path)
            img = image.convert('RGB')
            img = np.array(img)
            images.append(img)
        images = np.stack(images, axis=0)
        latent = np.load(os.path.join(
            self.latent_code_path, f'{uuid}.npz.npy'))
        latent = torch.tensor(latent)
        return {'caption': caption, 'latent': latent, 'image': images}


class Mydataset_val(Dataset):
    def __init__(self, latent_code_path):
        self.captions = [line.strip().split(',') for line in open(
            '/mnt/pfs/users/yuzhipeng/shapeE/shap-e/dataset/val.txt')]
        self.latent_code_path = latent_code_path

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        uuid, caption = self.captions[i]
        images = []
        for image_index in VIEW_INDEXS:
            # image_index = random.randint(1,24)
            image_path = f'{RENDER_PATH}/{uuid[:2]}/{uuid}/render_{image_index:03d}.png'
            image = Image.open(image_path)
            img = image.convert('RGB')
            img = np.array(img)
            images.append(img)
        images = np.stack(images, axis=0)
        latent = np.load(os.path.join(
            self.latent_code_path, f'{uuid}.npz.npy'))
        latent = torch.tensor(latent)
        return {'caption': caption, 'latent': latent, 'image': images}


def train(rank, args):
    if args.gpus > 1:
        setup_ddp(rank, args)

    niter = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    save_name = args.save_name
    f = open('./logs/%s.csv' % save_name, 'a')
    writer = csv.writer(f)
    torch.manual_seed(rank+int(learning_rate*1e6) +
                      int(datetime.now().timestamp()))
    resume_flag = True if args.resume_name != 'none' else False
    if resume_flag:
        model_list = glob.glob('./model_ckpts/%s*.pth' % save_name)
        idx_rank = []
        for l in model_list:
            idx_rank.append(int(l.split('/')[-1].split('_')[-2][5:]) * 21000 + int(
                l.split('/')[-1].split('_')[-1].split('.')[0]))
        newest = np.argmax(np.array(idx_rank))
        args.resume_name = model_list[newest].split('/')[-1].split('.')[0]

    start_epoch = 0 if not resume_flag else int(
        args.resume_name.split('_')[-2][5:])
    start_iter = 0 if not resume_flag else int(
        args.resume_name.split('_')[-1].split('.')[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if resume_flag:
        print('reload from ./model_ckpts/%s.pth' % args.resume_name)
        checkpoint = torch.load('./model_ckpts/%s.pth' %
                                args.resume_name, map_location=device)

    # if not resume_flag:
    #     model = load_model('image300M', device=device)
    # else:
    #     model = model_from_config(load_config_frompath(), device=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    if not resume_flag:
        model = load_model_from_path(
            'image300M', args.model_path,  device=device, config_path=args.model_config)
    else:
        model = load_model_from_path('image300M', './model_ckpts/%s.pth' %
                                     args.resume_name, device=device, config_path=args.model_config)
    print(model, flush=True)
    print('model loaded',flush=True)
    model.train()
    if args.gpus > 1:
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False
        )

    diffusion = diffusion_from_config(load_config('diffusion'))
    my_dataset = Mydataset(args.latent_code_path)
    # data_loader = DataLoader(my_dataset, batch_size=batch_size, num_workers=0, prefetch_factor=4, shuffle=True, drop_last=True)
    data_loader = DataLoader(
        my_dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True, )
    my_dataset_val = Mydataset_val(args.latent_code_path)
    # data_loader_val = DataLoader(my_dataset_val, batch_size=batch_size, num_workers=0, prefetch_factor=4, drop_last=True)
    data_loader_val = DataLoader(
        my_dataset_val, batch_size=batch_size, num_workers=0, drop_last=True, )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_iter_per_epoch = int(len(my_dataset)/batch_size)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, niter*total_iter_per_epoch)
    if resume_flag:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    for epoch in range(start_epoch, niter):
        s = time.time()
        for i, data in enumerate(data_loader):
            if i + start_iter == total_iter_per_epoch:
                start_iter = 0
                break
            s2 = time.time()
            prompt = data['caption']
            img_prompt = data['image']
            img_prompt_shape = img_prompt.shape
            img_prompt = img_prompt.reshape((batch_size*img_prompt_shape[1], img_prompt_shape[2], img_prompt_shape[3], img_prompt_shape[4]))

            print(f'{img_prompt.shape}, flush=True')

            model_kwargs = dict(images=img_prompt)
            t = torch.randint(0, load_config('diffusion')[
                              'timesteps'], size=(batch_size,), device=device)
            x_start = data['latent'].cuda()

            optimizer.zero_grad()
            # import ipdb; ipdb.set_trace()
            # print(img_prompt.shape, x_start.shape, t.shape, prompt, flush=True)
            loss = diffusion.training_losses(
                model, x_start, t, model_kwargs=model_kwargs)
            final_loss = torch.mean(loss['loss'])

            skip_step = torch.isnan(
                final_loss.detach()) or not torch.isfinite(final_loss.detach())
            skip_step_tensor = torch.tensor(
                skip_step, dtype=torch.int).to(device)
            if args.gpus > 1:
                dist.all_reduce(skip_step_tensor, op=dist.ReduceOp.SUM)
            skip_step = skip_step_tensor.item() > 0
            if skip_step:
                del final_loss
                torch.cuda.empty_cache()
            else:
                final_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                if args.gpus == 1 or (args.gpus > 1 and dist.get_rank() == 0):
                    print('rank: ', rank, time.time()-s2,
                          ' epoch: ', epoch, i, final_loss.item())
                if (epoch+1) % 50 == 0 and i==0:
                    if args.gpus > 1:
                        torch.save({'model_state_dict': model.module.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': lr_scheduler.state_dict(),
                                    }, './model_ckpts/%s_epoch%d_%d.pth' % (save_name, epoch, i+start_iter))
                    else:
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': lr_scheduler.state_dict(),
                                    }, './model_ckpts/%s_epoch%d_%d.pth' % (save_name, epoch, i+start_iter))
                if epoch % 50 == 0 and i==0:
                    with torch.no_grad():
                        val_loss = []
                        for j, dataval in enumerate(data_loader_val):
                            prompt = data['caption']
                            img_prompt = data['image']
                            img_prompt_shape = img_prompt.shape
                            img_prompt = img_prompt.reshape((batch_size*img_prompt_shape[1], img_prompt_shape[2], img_prompt_shape[3], img_prompt_shape[4]))
                            # model_kwargs=dict(texts=prompt)
                            model_kwargs = dict(images=img_prompt)
                            t = torch.randint(0, load_config('diffusion')[
                                              'timesteps'], size=(batch_size,), device=device)
                            x_start = data['latent'].cuda()
                            loss = diffusion.training_losses(
                                model, x_start, t, model_kwargs=model_kwargs)
                            final_loss = torch.mean(loss['loss'])
                            print('validation %d/%d: ' %
                                  (j, len(data_loader_val)), final_loss.item())
                            val_loss.append(final_loss.item())
                        val_mean_loss = torch.mean(
                            torch.Tensor(val_loss)).item()
                        writer.writerow([epoch, i+start_iter, val_mean_loss])
                        f.flush()
                        os.fsync(f.fileno())
                        print('rank: ', rank, i, val_mean_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_group = parser.add_argument_group('Model settings')
    model_group.add_argument(
        '--port', type=str, default='12356', help='port for parallel')
    model_group.add_argument('--random_view', action='store_true',)
    model_group.add_argument(
        '--gpus', type=int, default=1, help='how many gpu use')
    model_group.add_argument('--resume_name', type=str, default='none',
                             help='any name different from "none" will resume the training')
    model_group.add_argument(
        '--save_name', type=str, default='shape-image', help='name for the save file')
    model_group.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    model_group.add_argument('--batch_size', type=int,
                             default=64, help='batch size')
    model_group.add_argument(
        '--epoch', type=int, default=1000, help='total epoch')
    model_group.add_argument('--latent_code_path', type=str, default='/mnt/pfs/users/yuzhipeng/shapeE/shap-e/spaceship/all_latent',
                             help='the directory to the .pt file which store Shap-E latent codes')
    model_group.add_argument('--model_config', type=str,
                             default='/mnt/pfs/users/yuzhipeng/shapeE/shap-e/experiments/4view/image_cond_config.yaml', help='')
    model_group.add_argument(
        '--model_path', type=str, default='/mnt/pfs/users/yuzhipeng/shapeE/shap-e/pretrains/image_cond_4view.pt', help='')

    args = parser.parse_args()

    if args.gpus == 1:
        train(args.gpus, args)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        args.world_size = args.gpus
        mp.spawn(train, nprocs=args.gpus, args=(args,))
