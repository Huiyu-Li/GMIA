"""
Task: Latent Code Optimization with ID and UT losses
"""

import os, time
import click
import re
import json

# Custom modules
import optimization_loop
import sys
sys.path.append('/home/huili/Projects/my_coES_stylegan2/')
import dnnlib

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    seed       = None, # Random seed: <int>, default = 0
    vis_env    = None, # Random seed: <int>, default = 0

    # Dataset.
    csv        = None, # Training csv (required): <path>
    data       = None, # Training dataset: <path>
    semantic   = None, # Training sementic features: <path>
    latent     = None, # Training latent codes: <path>
    mode       = None, # Data split: 'train', 'valid', 'test'
    start      = None, # Start idx of train split
    end        = None, # End idx of train split

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'demo', 'paper256'
    # w_plus     = None, # Optimize in the W+ space.
    id_margin  = None, # Margin of identity loss.
    lambda_id  = None, # Coefficient of identity loss.
    lambda_ut  = None, # Coefficient of utility loss.
    # lambda_w   = None, # Coefficient of w loss.
    # w_cosine   = None, # Perform w cosine distance.
    # w_l2norm   = None, # Perform w l2norm distance.
    # w_il2norm  = None, # Perform w inverse l2norm distance.

    # Loss.
    id_ckpt    = None, # Pre-trained idenity checkpoint
    ut_ckpt    = None, # Pre-trained utility checkpoint
    g_ckpt     = None, # Pre-trained generator checkpoint

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, seed, visdom
    # ------------------------------------------
    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    assert vis_env is not None
    tag = g_ckpt.split('/')[-2]
    if start is not None and end is not None:
        vis_env = f'optimization {vis_env}: {tag}_{mode} [{start}:{end}]'
    else:
        vis_env = f'optimization {vis_env}: {tag}_{mode}'
    args.vis_kwargs = dnnlib.EasyDict(env=vis_env, server='http://nef-devel2', port=2024)

    # -----------------------------------
    # Dataset: data, cond
    # -----------------------------------
    if csv is None:
        csv = "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-split.csv"
    assert isinstance(csv, str)
    if data is None:
        data = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/files"
        resolution = 256
    if semantic is None:
        semantic = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/semantics/"
    if latent is None:
        latent = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/w_00023/"
    assert isinstance(data, str)
    assert mode is not None
    if mode == 'train':
        # start_end = [0: 12719] [12719: 25438] [25438: 38157] [38157: 50876]
        print(f'start:end = [{start}: {end}]')
        args.dataset_kwargs = dnnlib.EasyDict(class_name='src.dataset.MIMIC_Latent', 
                        csv_path=csv, image_root=data, semantic_root=semantic, latent_root=latent, 
                        mode=mode, start=start, end=end, use_transform=False, shuffle=False, load_z=True)
    else:
        args.dataset_kwargs = dnnlib.EasyDict(class_name='src.dataset.MIMIC_Latent', 
                    csv_path=csv, image_root=data, semantic_root=semantic, latent_root=latent, 
                    mode=mode, use_transform=False, shuffle=False, load_z=True)
    
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    desc = 'MIMIC_Latent'
        
    # ------------------------------------
    # Base config: kimg, batch
    # ------------------------------------
    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto': dict(run_epochs=1000,  early_stop=1e-8,  batch_size=1,  id_margin=-0.5, lambda_id=1,  lambda_ut=1,  fmaps=0.5),
        'demo': dict(run_epochs=1000,  early_stop=1e-8,  batch_size=1,  id_margin=-0.5, lambda_id=1,  lambda_ut=1,  fmaps=0.5),
    }
    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    args.run_epochs=spec.run_epochs; args.early_stop=spec.early_stop
    args.batch_size=spec.batch_size; 
    desc += f'-epoch{args.run_epochs}'
    if args.batch_size > 1:
        desc += f'-batch{args.batch_size}'
    
    # args.w_plus = w_plus
    if id_margin is None:
       id_margin=spec.id_margin

    if lambda_id is None:
        args.lambda_id=spec.lambda_id
    else:
        args.lambda_id=lambda_id

    if lambda_ut is None:
        args.lambda_ut=spec.lambda_ut
    else:
        args.lambda_ut=lambda_ut
    desc += f'-margin{str(id_margin)}-id{str(lambda_id)}-ut{str(lambda_ut)}'
    # if lambda_w is None:
    #     args.lambda_w=spec.lambda_w
    # else:
    #     args.lambda_w=lambda_w
    # args.w_cosine = w_cosine; args.w_l2norm=w_l2norm; args.w_il2norm=w_il2norm

    args.W_opt_kwargs = dnnlib.EasyDict(class_name = 'torch.optim.Adam', #['adam', 'adamw']
                        lr = 0.01, # [1e-4(default: 1e-3), 3e-5(default: 1e-3)]
                        betas=[0,0.99],
                        eps = 1e-8, #  [1e-6(default: 0), 1e-2(default)]
                        )
    
    if id_ckpt is None:
        # notrans r50 1channel
        id_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r50_2024_02_07_12_20/checkpoints/r50_epoch299_idx105900_2024_02_08_04:32:41.pt"
    if ut_ckpt is None:
        # ChecXclusion uDense DenseNet121 pretrained (scheduler SoftLabel-1) 
        ut_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_02_13_22_24/checkpoints/Step2_MIMIC_Utility_DenseNet121_2024_02_13_22_24_epoch99_idx105900_2024_02_14_16:14.pt"
    args.id_loss_kwargs = dnnlib.EasyDict(id_margin=id_margin, network="r50", in_channel=1, embedding_size=512, id_ckpt=id_ckpt)
    args.ut_loss_kwargs = dnnlib.EasyDict(input_nc=3, normalize=True, pretrained=False, activations='elu', 
                                          last_activation=None, num_classes=4, ut_ckpt=ut_ckpt)
    
    assert g_ckpt is not None
    args.G_ckpt = g_ckpt; args.num_ws=14
    args.G_kwargs = dnnlib.EasyDict(class_name='src.network.StyleGAN2.ESModel', z_dim=512, c_dim=0, w_dim=512, 
                                    img_resolution=resolution, img_channels=1,
                                    synthesis_kwargs=dnnlib.EasyDict(channel_base=int(spec.fmaps * 32768), 
                                                                     channel_max=512, num_fp16_res=4, conv_clamp=256))

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------
    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        common_kwargs = dict(num_fp16_res=0, conv_clamp=None)
        args.G_kwargs.synthesis_kwargs.update(common_kwargs)

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = True

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
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('--vis_env', help='Visdom environment name', type=str, metavar='ENV')

# Dataset.
@click.option('--csv', help='Training data csv file', type=str, metavar='CSV')
@click.option('--data', help='Training data directory', type=str, metavar='PATH')
@click.option('--semantic', help='Training sementic directory', type=str, metavar='PATH')
@click.option('--latent', help='Training latent directory', type=str, metavar='PATH')
@click.option('--mode', help='Data split', type=click.Choice(['train', 'valid', 'test']))
@click.option('--start', help='Start idx of train split', type=int, metavar='INT')
@click.option('--end', help='End idx of train split', type=int, metavar='INT')

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'demo']))
# @click.option('--w_plus', help='Optimization in W+ latent space', type=bool, metavar='BOOL')
@click.option('--id_margin', help='Margin of identity loss.', type=float, metavar='Float')
@click.option('--lambda_id', help='Coefficient of identity loss.', type=float, metavar='Float')
@click.option('--lambda_ut', help='Coefficient of utility loss.', type=float, metavar='Float')
# @click.option('--lambda_w', help='Coefficient of w loss.', type=int, metavar='INT')
# @click.option('--w_cosine', help='Perform w cosine distance.', type=bool, metavar='BOOL')
# @click.option('--w_l2norm', help='Perform w l2norm distance.', type=bool, metavar='BOOL')
# @click.option('--w_il2norm', help='Perform w inverse l2norm distance.', type=bool, metavar='BOOL')

# Loss.
@click.option('--id_ckpt', help='Pre-trained idenity checkpoint', metavar='PKL')
@click.option('--ut_ckpt', help='Pre-trained utility checkpoint', metavar='PKL')
@click.option('--g_ckpt', help='Pre-trained generator checkpoint', metavar='PKL')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir=None, **config_kwargs):

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
    # cur_run_id = 38
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')

    # Print options.
    # print()
    # print('Training options:')
    # print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.dataset_kwargs.image_root}')
    print(f'Training data:      {args.dataset_kwargs.semantic_root}')
    print(f'Training data:      {args.dataset_kwargs.latent_root}')
    # print(f'Number of images:   {args.dataset_kwargs.max_size}')
    # print(f'Image resolution:   {resolution}')
    print()

    # Create output directory.
    print('Creating output directory...')
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
        os.makedirs(os.path.join(args.run_dir, 'anony'))
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
         json.dump(args, f, indent=2)

    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    # Execute training loop.
    optimization_loop.training_loop(**args)

    print(f'Time elapsed:  {((time.time() - start_time) / 60):.3f} min')
    print(f"End time:      {time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime())}")
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
