# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import json
import copy
import torch
import pickle
import piq
import pandas as pd
from torchvision.utils import save_image
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("/home/huili/Projects/my_coES_stylegan2")
import dnnlib

#----------------------------------------------------------------------------

def evaluation_loop(args):
    start_time = time.time()
    dnnlib.util.Logger(should_flush=True)

    # Print network summary.
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)

    # DataLoader
    dataset = dnnlib.util.construct_class_by_name(**args.dataset_kwargs)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, **args.data_loader_kwargs)

    # metrics
    n, N = 0, len(data_loader)
    data_range = 1.
    criteria_list =['PSNR↑', 'SSIM↑', 'MS_SSIM↑', 'IW_SSIM↑', 'LPIPS↓'] # Need to test the best value

    criteria = {}
    for name in criteria_list:
        criteria[name] = torch.Tensor(N)

    # Main loop.
    for fname, images, _labels in data_loader:
        real_img = images.to(device, dtype=torch.float32)
        gen_img = G(real_img, _labels, noise_mode=args.noise_mode)
        gen_img = torch.clamp(gen_img, min=0., max=1.)

        # metrics
        criteria['PSNR↑'][n:n + 1] = piq.psnr(real_img, gen_img, data_range, reduction='mean')
        
        criteria['SSIM↑'][n:n + 1] = piq.ssim(real_img, gen_img, data_range)
        criteria['MS_SSIM↑'][n:n + 1] = piq.multi_scale_ssim(real_img, gen_img, data_range)
        criteria['IW_SSIM↑'][n:n + 1] = piq.information_weighted_ssim(real_img, gen_img, data_range)

        criteria['LPIPS↓'][n:n + 1] = piq.LPIPS(reduction='mean')(real_img, gen_img)

        n += 1
        # save sub-output
        # for i in range(args.batch_size):
        #     save_dir = os.path.join(args.reco_root, fname[i][:-4].replace('/','_'))
        #     # save_dir = os.path.join(args.reco_root, fname[i][:-4])
        #     # utils.makedirs(save_dir)
        #     save_image(real_img[i,0], fp=f'{save_dir}_R.jpg')
        #     save_image(gen_img[i,0], fp=f'{save_dir}_A.jpg')

    # metrics
    df = pd.DataFrame(criteria)

    # metrics_to_avg_std_latex_table    
    latex_str = ''
    for key in criteria_list:
        avg = df.loc[:,key].mean()
        std = df.loc[:,key].std()
        latex_str += f'&{avg:.3f}$\pm${std:.3f} '
    print(latex_str)

    # metrics_to_avg_std_table
    criteria = {}
    for key in criteria_list:
        avg = df.loc[:,key].mean()
        std = df.loc[:,key].std()
        criteria[key] = f'{avg:.3f}±{std:.3f}' 
    print(pd.DataFrame(criteria, index=[0]))

    jsonl_line = json.dumps(dict(criteria, network_pkl=args.network_pkl, timestamp=time.time()))
    with open(os.path.join(args.outdir, f'eval.jsonl'), 'at') as f:
        f.write(jsonl_line + '\n')

    # Done.
    print('Time {:.3f} min'.format((time.time() - start_time) / 60))
    print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))

#----------------------------------------------------------------------------
def calc_metrics(    
    # General options (not included in desc).
    network_pkl    = None, # Network pickle filename or URL
    G_key          = 'G_ema', # type=click.Choice(['G_ema', 'G'])
    noise_mode     = 'const', # type=click.Choice(['const', 'random', 'none'])
    outdir         = None,
    verbose        = True, # Print optional information
    
    # Dataset.
    csv            = None, # Training csv (required): <path>
    data           = None, # Training dataset (required): <path>
    ):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=pr50k3_full \\
        --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    Available metrics:

    \b
      ADA paper:
        fid50k_full  Frechet inception distance against the full dataset.
        kid50k_full  Kernel inception distance against the full dataset.
        pr50k3_full  Precision and recall againt the full dataset.
        is50k        Inception score for CIFAR-10.

    \b
      StyleGAN and StyleGAN2 papers:
        fid50k       Frechet inception distance against 50k real images.
        kid50k       Kernel inception distance against 50k real images.
        pr50k3       Precision and recall against 50k real images.
        ppl2_wend    Perceptual path length in W at path endpoints against full image.
        ppl_zfull    Perceptual path length in Z for full paths against cropped image.
        ppl_wfull    Perceptual path length in W for full paths against cropped image.
        ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
        ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(network_pkl=network_pkl, 
                           noise_mode=noise_mode, outdir=outdir, verbose=verbose)
    
    # Load network.
    print('Loading networks from "%s"...' % network_pkl)
    with open(network_pkl, 'rb') as f:
        network_dict = pickle.load(f)
    # Validate contents.
    assert isinstance(network_dict[G_key], torch.nn.Module)
    args.G  = network_dict[G_key] # type: ignore

    # Set dataset arguments  
    if csv is None:
        csv = "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-split.csv"
    assert isinstance(csv, str)
    if data is None:
        data = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/files" 
    assert isinstance(data, str)
    args.dataset_kwargs = dnnlib.EasyDict(class_name='src.dataset.MIMIC', 
                    csv_path=csv, image_root=data, mode='test', oui = [True, False, False],
                    img_channels=1, use_labels=False)
    args.data_loader_kwargs = dnnlib.EasyDict(batch_size=64, shuffle=False, num_workers=3, prefetch_factor=2, collate_fn=None, pin_memory=True, drop_last=True)
    
    # Print dataset options.
    # if args.verbose:
    #     print('Dataset options:')
    #     print(json.dumps(args.dataset_kwargs, indent=2))

    # Launch processes.
    evaluation_loop(args=args)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    # Variables
  save_dir = "/data/epione/user/huili/exp_coES/evaluation"
  network_pkl_list={
    # E-S Co-training (without S_reg)
    # "/data/epione/user/huili/exp_coES/00008-MIMIC-auto25k-coG/network-snapshot-018144.pkl"
    # E-S Co-training (with S_reg)
    # "/data/epione/user/huili/exp_coES/00009-MIMIC-auto25k-Greg-coG/network-snapshot-014918.pkl"

    # E-S-D Co-training (without self-pretrained E-S)
    # "/data/epione/user/huili/exp_coES/00010-MIMIC-auto25k-Greg-coGD/network-snapshot-006249.pkl",  
    # "/data/epione/user/huili/exp_coES/00011-MIMIC-auto25k-Greg-coGD-resume/network-snapshot-007056.pkl",  
    # "/data/epione/user/huili/exp_coES/00013-MIMIC-auto25k-Greg-coGD-resume/network-snapshot-007056.pkl", 
    # "/data/epione/user/huili/exp_coES/00023-MIMIC-auto25k-Greg-coGD/network-snapshot-006854.pkl",
    # "/data/epione/user/huili/exp_coES/00028-MIMIC-auto25k-Greg-coGD-resume/network-snapshot-005644.pkl",
    # "/data/epione/user/huili/exp_coES/00028-MIMIC-auto25k-Greg-coGD-resume/network-snapshot-006854.pkl",
    # "/data/epione/user/huili/exp_coES/00045-MIMIC-Reco-auto25k-Greg-coGD/network-snapshot-006963.pkl"
    # "/data/epione/user/huili/exp_coES/00055-MIMIC-Reco-auto25k-Greg-coGD/network-snapshot-024576.pkl",

    # E-S-D Co-training (with pretrained E-S)
    # "/data/epione/user/huili/exp_coES/00016-MIMIC-auto25k-Greg-coGD-resumeG/network-snapshot-006854.pkl", 
    # "/data/epione/user/huili/exp_coES/00017-MIMIC-auto25k-Greg-coGD-resumeG/network-snapshot-007056.pkl", 
    # "/data/epione/user/huili/exp_coES/00022-MIMIC-auto25k-Greg-coGD-resumeG/network-snapshot-003225.pkl",
    # "/data/epione/user/huili/exp_coES/00022-MIMIC-auto25k-Greg-coGD-resumeG/network-snapshot-007056.pkl",
    }
  
  for network_pkl in network_pkl_list:
    assert os.path.isfile(network_pkl)
    outdir = os.path.join(save_dir, network_pkl.split('/')[-2])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # noise_mode_list = ['const', 'random', 'none']
    # for noise_mode in noise_mode_list:
    noise_mode='const'
    G_key = 'G_ema'

    calc_metrics(network_pkl, G_key, noise_mode, outdir)

#----------------------------------------------------------------------------
