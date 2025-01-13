# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
from typing import Optional
from torchvision.utils import save_image, make_grid
import pickle
import numpy as np
import PIL.Image
import torch

import sys
sys.path.append("/home/huili/Projects/my_coES_stylegan2")
#----------------------------------------------------------------------------

def generate_images(
    fnames: list,
    network_pkl: str,
    G_key: str='G_ema', # type=click.Choice(['G_ema', 'G'])
    noise_mode: str='const', # type=click.Choice(['const', 'random', 'none'])
    outdir: str=None,
    class_idx: Optional[int]=None,
    projected_w: Optional[str]=None # 'Projection result file'
):
    device = torch.device('cuda')
    
    print('Loading networks from "%s"...' % network_pkl)
    with open(network_pkl, 'rb') as f:
        resume_data = pickle.load(f)
    # Validate contents.
    assert isinstance(resume_data[G_key], torch.nn.Module)
    G = resume_data[G_key].eval().requires_grad_(False).to(device)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'L').save(f'{outdir}/proj{idx:02d}.png')
        return

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            print('error: Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    from PIL import Image
    from torchvision import transforms
    image_root = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/files"
    
    real_images = []
    fake_images = []
    for fname in fnames:
      image_path = f'{image_root}/{fname}'
      transform = transforms.Compose([transforms.ToTensor()])
      real_img = Image.open(image_path).convert('L') # PIL.Image.Image, [Columns, Rows]
      real_img = transform(real_img)
      real_img = real_img.unsqueeze(0)
      real_img = real_img.to(device, dtype=torch.float32)
      gen_img = G(real_img, label, noise_mode=noise_mode)
      
      real_images.append(real_img)
      fake_images.append(gen_img)

      # print('Saving images...')
      # fp_R = f'{outdir}/X_R_{fname[-8:]}'
      # if not os.path.isfile(fp_R):
      #    save_image(real_img, fp=fp_R)
      # save_image(gen_img, fp=f'{outdir}/X_A_{G_key}_{noise_mode}_{fname[-8:]}')
      # # save_image(img.clamp(0, 1), fp=f'{outdir}/seed{seed:04d}_clamp.png')
      # # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
      # # PIL.Image.fromarray(img[0].cpu().numpy(), 'L').save(f'{outdir}/seed{seed:04d}.png')

    print('Saving image grid...')
    images = torch.cat(real_images+fake_images, 0)
    grid = make_grid(images, nrow=len(fnames))
    save_image(grid, fp=f'{outdir}.png')
#----------------------------------------------------------------------------

if __name__ == "__main__":
  fnames = [
    # From TrainSet
    'p10/p10001884/s52060840/ee31086f-cbf22f9d-9553d506-2bcd4167-0e1e17bf.jpg',
    'p10/p10003400/s58983613/78b25180-ff33317d-92a10132-dffdc0b1-e7234a43.jpg',
    # 'p10/p10018081/s52153377/6bc14657-810b05e0-4bd32106-c30afa91-77f0122c.jpg',
    # # From ValidSet
    # 'p11/p11717909/s51345024/74ada62d-569c8df3-d20cc6c4-27858ab1-6bf22d69.jpg',
    'p11/p11888614/s50536002/3f69336f-36ceec41-467c3490-22a37536-b48f30e3.jpg',
    'p13/p13421580/s51827027/312bb0ed-2dafb619-a0da3729-5dc19055-53169588.jpg',
    # From TestSet
    # 'p19/p19016834/s55946640/ed9628e5-62ce1427-67e04f11-6daf5632-424ef2d1.jpg',
    'p14/p14295224/s55257496/7fb0f54f-a18826e9-05962b2b-66a603ac-a0991889.jpg',
    'p15/p15114531/s53033654/92d9fd50-81412806-b71e4d05-9ef38071-6b25204c.jpg',
  ]

  # Variables
  save_dir = "/data/epione/user/huili/exp_coES/generate"
  network_pkl_list={
    # E-S Co-training (without S_reg)
    # "/data/epione/user/huili/exp_coES/00008-MIMIC-auto25k-coG/network-snapshot-018144.pkl",   
    # "/data/epione/user/huili/exp_coES/00011-MIMIC-auto25k-coG-resume/network-snapshot-017942.pkl",  
    # "/data/epione/user/huili/exp_coES/00015-MIMIC-auto25k-coG-resume/network-snapshot-017942.pkl",  

    # "/data/epione/user/huili/exp_coES/00005-MIMIC-auto1-coG/network-snapshot-005088.pkl",  
    # E-S Co-training (with S_reg)
    # "/data/epione/user/huili/exp_coES/00009-MIMIC-auto25k-Greg-coG/network-snapshot-014918.pkl",   
    # "/data/epione/user/huili/exp_coES/00012-MIMIC-auto25k-Greg-coG-resume/network-snapshot-014313.pkl",   
    # "/data/epione/user/huili/exp_coES/00014-MIMIC-auto25k-Greg-coG-resume/network-snapshot-014313.pkl",  

    # "/data/epione/user/huili/exp_coES/00006-MIMIC-auto1-coG-Greg/network-snapshot-005040.pkl",  

    # E-S-D Co-training (without self-pretrained E-S)
    # "/data/epione/user/huili/exp_coES/00010-MIMIC-auto25k-Greg-coGD/network-snapshot-006249.pkl",  
    # "/data/epione/user/huili/exp_coES/00011-MIMIC-auto25k-Greg-coGD-resume/network-snapshot-007056.pkl",  
    # "/data/epione/user/huili/exp_coES/00013-MIMIC-auto25k-Greg-coGD-resume/network-snapshot-007056.pkl", 
    # "/data/epione/user/huili/exp_coES/00023-MIMIC-auto25k-Greg-coGD/network-snapshot-006854.pkl",
    # "/data/epione/user/huili/exp_coES/00028-MIMIC-auto25k-Greg-coGD-resume/network-snapshot-005644.pkl",
    # "/data/epione/user/huili/exp_coES/00028-MIMIC-auto25k-Greg-coGD-resume/network-snapshot-006854.pkl",
    # "/data/epione/user/huili/exp_coES/00045-MIMIC-Reco-auto25k-Greg-coGD/network-snapshot-006963.pkl"
    # "/data/epione/user/huili/exp_coES/00055-MIMIC-Reco-auto25k-Greg-coGD/network-snapshot-024576.pkl",

    # "/data/epione/user/huili/exp_coES/00007-MIMIC-auto1-coGD/network-snapshot-005088.pkl", 
    # "/data/epione/user/huili/exp_coES/00003-MIMIC-auto1-coGD/network-snapshot-005088.pkl", 
    # "/data/epione/user/huili/exp_coES/00004-MIMIC-auto1-coGD-noNoise/network-snapshot-005088.pkl", 

    # "/data/epione/user/huili/exp_coES/00001-MIMIC-auto1-coGD/network-snapshot-001612.pkl",
    # "/data/epione/user/huili/exp_coES/00002-MIMIC-auto1-coGD-resume/network-snapshot-005088.pkl", 

    # E-S-D Co-training (with pretrained E-S)
    # "/data/epione/user/huili/exp_coES/00016-MIMIC-auto25k-Greg-coGD-resumeG/network-snapshot-006854.pkl", 
    # "/data/epione/user/huili/exp_coES/00017-MIMIC-auto25k-Greg-coGD-resumeG/network-snapshot-007056.pkl", 
    # "/data/epione/user/huili/exp_coES/00022-MIMIC-auto25k-Greg-coGD-resumeG/network-snapshot-003225.pkl",
    # "/data/epione/user/huili/exp_coES/00022-MIMIC-auto25k-Greg-coGD-resumeG/network-snapshot-007056.pkl",

  }
  
  for network_pkl in network_pkl_list:
    assert os.path.isfile(network_pkl)
    outdir = os.path.join(save_dir, network_pkl.split('/')[-2])
    # noise_mode_list = ['const', 'random', 'none']
    # for noise_mode in noise_mode_list:
    noise_mode='const'
    G_key = 'G_ema'
    generate_images(fnames, network_pkl, G_key, noise_mode, outdir)