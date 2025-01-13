# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Task: StyleGAN2 inversion in W+ space
Optimization: G(E-S) with shared optimizer, D with idependant optimizer
Approach:
Co-training for Generator (including a symmetric Encoder with a Synthesis decoder) on custom dataset.
Co-training for Generator (including a symmetric Encoder with a Synthesis decoder) and Descriminator on custom dataset.
"""

import os, time
import click
import re
import json
import tempfile
import torch

# Custom modules
import reco_loop
import dnnlib
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    seed       = None, # Random seed: <int>, default = 0

    # Dataset.
    csv        = None, # Training csv (required): <path>
    data       = None, # Training dataset (required): <path>
    cond       = None, # Train conditional model based on dataset labels: <bool>, default = False

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'demo', 'paper256'
    cogd       = None, # Enable co-training of G(E-S), D network
    g_reg      = None, # Enable regularization for G
    gamma      = None, # Override R1 gamma: <float>
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>

    # Transfer learning.
    resume_g   = None, # Load previous G network: <file>
    resume     = None, # Load previous network: 'noresume' (default), 'resumeG', <file>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, cond
    # -----------------------------------
    if csv is None:
        csv = "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-split.csv"
    assert isinstance(csv, str)
    if data is None:
        data = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/files" 
        args.resolution=256
    assert isinstance(data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='src.dataset.MIMIC', 
                        csv_path=csv, image_root=data, mode='train', img_channels=1,
                        use_transform=True,normalize=False, 
                        oui = [True, False, False])
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    desc = 'MIMIC-Reco'
    # try:
        # training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)
        # args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        # desc = 'MIMIC'# training_set.name
        # del training_set # conserve memory
    # except IOError as err:
    #     raise UserError(f'--data: {err}')
        
    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------
    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'demo':      dict(ref_gpus=1,  kimg=5088,   mb=128, mbstd=4,  fmaps=0.5, lrate=0.0025, mse=10, perceptual=10, gamma=-1,   ema=-1,  ramp=0.05, map=8), # Populated dynamically based on resolution and GPU count.
        'auto25k':   dict(ref_gpus=1,  kimg=25000,  mb=128, mbstd=4,  fmaps=0.5, lrate=0.0025, mse=10, perceptual=10, gamma=1,    ema=20,  ramp=None, map=8),
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])

    if gpus>1 and spec.ref_gpus != gpus:
        spec.ref_gpus = gpus

    if cfg == 'auto' or cfg == 'demo':
        desc += f'{gpus:d}'
        # spec.ref_gpus = gpus
        res = args.resolution
        # spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        # spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        # spec.fmaps = 1 if res >= 512 else 0.5
        # spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32
        print(f'===>spec.gamma:{spec.gamma}, spec.ema:{spec.ema}')

    if cogd is None:
        cogd = False
    if g_reg is None:
        args.G_reg_interval=None
    else:
        desc += '-Greg'
        args.G_reg_interval=4

    args.G_kwargs = dnnlib.EasyDict(class_name='src.network.StyleGAN2.ESModel', z_dim=512, w_dim=512, 
                                    synthesis_kwargs=dnnlib.EasyDict())
    common_kwargs = dict(channel_base=int(spec.fmaps * 32768), channel_max=512, num_fp16_res=4, conv_clamp=256)
    # num_fp16_res: enable mixed-precision training
    # conv_clamp: clamp activations to avoid float16 overflow
    args.G_kwargs.synthesis_kwargs.update(common_kwargs)
    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
   
    if cogd:
        desc += '-coGD'
        args.coGD = True
        args.D_kwargs = dnnlib.EasyDict(class_name='src.network.StyleGAN2.Discriminator', 
                                block_kwargs=dnnlib.EasyDict(), 
                                mapping_kwargs=dnnlib.EasyDict(), 
                                epilogue_kwargs=dnnlib.EasyDict())
        args.D_kwargs.update(common_kwargs)
        args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
        args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        args.loss_kwargs = dnnlib.EasyDict(class_name='src.loss.StyleGAN2.ESD_CoTrainingLoss', 
                                           lambda_mse=spec.mse, lambda_perceptual=spec.perceptual, r1_gamma=spec.gamma)
    else:
        desc += '-coG'
        args.loss_kwargs = dnnlib.EasyDict(class_name='src.loss.StyleGAN2.ES_CoTrainingLoss', 
                                           lambda_mse=spec.mse, lambda_perceptual=spec.perceptual, r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    print(f'===>batch_size: {args.batch_size}, batch_gpu:{args.batch_gpu}, ref_gpus:{spec.ref_gpus}')
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------
    assert resume is None or isinstance(resume, str)
    assert (resume and resume_g) is None
    if resume is None or resume == 'noresume':
        resume = 'noresume'
    else:
        desc += '-resume'
        args.resume_pkl = resume # custom path or url

    if resume_g:
        desc += '-resumeG'
        args.resume_G = resume_g # custom path or url

    if (resume != 'noresume') or (resume_g is not None):
        args.ema_rampup = None # disable EMA rampup
        # if ema_rampup=None ema_beta=0.9977843871238888, 
        # otherwise ema_beta increased progressively

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        common_kwargs = dict(num_fp16_res=0, conv_clamp=None)
        args.G_kwargs.synthesis_kwargs.update(common_kwargs)
        if cogd:
            args.D_kwargs.update(common_kwargs)

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = True
        if cogd:
            args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        init_method = f'file://{init_file}'
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    reco_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', type=str, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')

# Dataset.
@click.option('--csv', help='Training data csv file', type=str, metavar='CSV')
@click.option('--data', help='Training data directory', type=str, metavar='PATH')
@click.option('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, metavar='BOOL')

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'auto25k', 'demo', 'paper256']))
@click.option('--cogd', help='Enable co-training of G(E-S), D network', type=bool, metavar='BOOL')
@click.option('--g_reg', help='Enable regularization for G', type=bool, metavar='BOOL')
@click.option('--gamma', help='Override R1 gamma', type=float)
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--batch', help='Override batch size', type=int, metavar='INT')

# Transfer learning.
@click.option('--resume_g', help='Resume training G', metavar='PKL')
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')
@click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir=None, **config_kwargs):
    """
    Co-training for a Symmetric Encoder with a StyleGAN2 on custom dataset.

    Examples:
    """
    start_time = time.time()
    print(f"Start Time:  {time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime())}")

    dnnlib.util.Logger(should_flush=True)
    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    if outdir is None:
        outdir = "/data/epione/user/huili/exp_coES"
    assert isinstance(outdir, str)
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    # print()
    # print('Training options:')
    # print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.image_root}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    # print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    print()

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    os.makedirs(os.path.join(args.run_dir, 'images'))
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
         json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

    print(f'Time elapsed:  {((time.time() - start_time) / 60):.3f} min')
    print(f"End time:      {time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime())}")
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
