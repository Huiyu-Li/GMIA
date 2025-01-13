'''
In eval mode,
Extract the id and ut semantic features 
and the w latent code from the best pre-trained checkpoint.
'''

import os, time
import pickle
import torch
import click

# Custom modules
import sys
sys.path.append('/home/huili/Projects/my_coES_stylegan2/')
import dnnlib
from src.network.iresnet import iresnet50
from src.network.densenet_utility import DenseNet121
from torch_utils.ops import conv2d_gradfix
#----------------------------------------------------------------------------

# Reproducibility
def make_deterministic(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False         # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = False # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = False       # Allow PyTorch to internally use tf32 for convolutions
        conv2d_gradfix.enabled = True                 # Improves training speed.
#----------------------------------------------------------------------------

def extraction_identity(device, data_loader, model_identity, semantic_root):
    with torch.no_grad():
        for fname, images, _labels in data_loader:
            real_img = images.to(device, dtype=torch.float32)
            # Get identity latent
            s_identity = model_identity(real_img)

            # save latents
            save_dir = os.path.join(semantic_root,fname[0][:-4])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            assert s_identity.shape[0]==1, 'N must be equal to 1.'
            fname_s = os.path.join(save_dir, 'S_id.pt')
            # print(f's_identity: {s_identity.shape}, fname_s:{fname_s}')
            torch.save(s_identity[0].detach().cpu(), fname_s)

#----------------------------------------------------------------------------

def extraction_utility(device, data_loader, model_utility, semantic_root):
    with torch.no_grad():
        for fname, images, _labels in data_loader:
            real_img = images.to(device, dtype=torch.float32)
            # Get utility latent
            s_utility = model_utility.encoder(real_img)

            # save latents
            save_dir = os.path.join(semantic_root,fname[0][:-4])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            assert s_utility.shape[0]==1, 'N must be equal to 1.'
            fname_s = os.path.join(save_dir, 'S_ut.pt')
            # print(f's_utility: {s_utility.shape}, fname_s:{fname_s}')
            torch.save(s_utility[0].detach().cpu(), fname_s)

#----------------------------------------------------------------------------

def extraction_latent(device, G, data_loader, latent_root): 
    with torch.no_grad(): 
        for fname, images, _labels in data_loader:
            real_img = images.to(device, dtype=torch.float32)
            gen_w = G.encoder(real_img, _labels) # [1, 14, 512]
            # print(f'gen_w: {gen_w.shape}')
            
            # save latents
            save_dir = os.path.join(latent_root,fname[0][:-4])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            assert gen_w.shape[0]==1, 'N must be equal to 1.'
            # Save in shape [L=512]
            fname_W = os.path.join(save_dir, 'W_ge.pt')
            # print(f'fname_W:{fname_W}')
            torch.save(gen_w[0,0].detach().cpu(), fname_W)

#----------------------------------------------------------------------------

@click.command()
# General options.
@click.option('--choice', help='Choice of the semantics or latent', type=click.Choice(['identity', 'utility', 'latent']))
@click.option('--mode', help='Data split', type=click.Choice(['train', 'valid', 'test']))
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
# Dataset.
@click.option('--csv', help='Training data csv file', type=str, metavar='CSV')
@click.option('--data', help='Training data directory', type=str, metavar='PATH')
@click.option('--semantic', help='Training sementic directory', type=str, metavar='PATH')
@click.option('--latent', help='Training latent directory', type=str, metavar='PATH')
# Model
@click.option('--id_ckpt', help='Pre-trained idenity checkpoint', metavar='PKL')
@click.option('--ut_ckpt', help='Pre-trained utility checkpoint', metavar='PKL')
@click.option('--g_ckpt', help='Pre-trained generator checkpoint', metavar='PKL')
@click.option('--g_key', help='Generator type', type=click.Choice(['G_ema', 'G']))

def main(
    # General options (not included in desc).
    choice     = None, # Data split: 'identity', 'utility', 'latent'
    mode       = None, # Data split: 'train', 'valid', 'test'
    seed       = None, # Random seed: <int>, default = 0

    # Dataset.
    csv        = None, # Training csv (required): <path>
    data       = None, # Training dataset: <path>
    semantic   = None, # Training sementic features: <path>
    latent     = None, # Training latent codes: <path>

    # Model
    id_ckpt    = None, # Identity network checkpoint
    ut_ckpt    = None, # Utility network checkpoint
    g_ckpt     = None, # Generator network checkpoint
    g_key      = None, # Generator type: 'G_ema', 'G'
):
    """
    Extract the id and ut semantic features 
    and the w latent code from the best pre-trained checkpoint.
    """
    start_time = time.time()
    dnnlib.util.Logger(should_flush=True) 
    if seed is None:
        seed = 0
    make_deterministic(seed)
    device = torch.device('cuda')

    # Set dataset arguments  
    if csv is None:
        csv = "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-split.csv"
    assert isinstance(csv, str)
    if data is None:
        data = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/files" 
        resolution = 256
    assert isinstance(data, str)

    if choice == 'utility':
         img_channels = 3
         normalize = True
    else:
         img_channels = 1
         normalize = False
    dataset_kwargs = dnnlib.EasyDict(class_name='src.dataset.MIMIC', 
                    csv_path=csv, image_root=data, mode=mode, 
                    img_channels=img_channels, use_transform=False, normalize=normalize,
                    use_labels=False, shuffle=False, oui = [True, False, False])

    print('Loading training set...')
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    data_loader_kwargs = dnnlib.EasyDict(batch_size=1, shuffle=False, num_workers=3, prefetch_factor=2, 
                                         collate_fn=None, pin_memory=True, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, **data_loader_kwargs)
    print(f'Num images of {mode}: {len(dataset)}')
    print('Image resolution:', resolution)

    # if semantic is None:
    #     semantic = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/semantics/"
    if choice == 'identity':
      assert semantic is not None
      if id_ckpt is None:
        # notrans r50 1channel
        id_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r50_2024_02_07_12_20/checkpoints/r50_epoch299_idx105900_2024_02_08_04:32:41.pt"

      # Loading Identity Encoder "r50"
      # More model choice refer to My_ArcFace foler
      backbone = iresnet50(in_channel=1, dropout=0.0, num_features=512)
      print('Loading identity network from "%s"...' % id_ckpt)
      dict_checkpoint = torch.load(id_ckpt)
      # dict_keys(['epoch', 'global_step', 'state_dict_backbone', 'state_dict_softmax_fc', 'state_optimizer', 'state_lr_scheduler'])
      backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
      del dict_checkpoint
      model_identity = backbone.eval().requires_grad_(False).to(device)
    
      extraction_identity(device, data_loader, model_identity, semantic)

    elif choice == 'utility':
        assert semantic is not None
        if ut_ckpt is None:
            # ChecXclusion uDense DenseNet121 pretrained (scheduler SoftLabel-1) 
            ut_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_02_13_22_24/checkpoints/Step2_MIMIC_Utility_DenseNet121_2024_02_13_22_24_epoch99_idx105900_2024_02_14_16:14.pt"

        # Loading Utiliyt Encoder
        model_utility = DenseNet121(num_in_features=3, pretrained=False, 
                            activations='elu', last_activation=None, num_classes=4)
        print('Loading utility network from "%s"...' % ut_ckpt)
        checkpoint_utility = torch.load(ut_ckpt)
        model_utility.load_state_dict(checkpoint_utility['model_state_dict'], strict=False)# `strict` False if warmstarting
        del checkpoint_utility
        model_utility = model_utility.eval().requires_grad_(False).to(device)

        extraction_utility(device, data_loader, model_utility, semantic)
    
    elif choice == 'latent':
        # assert g_ckpt is not None
        if g_ckpt is None:
           g_ckpt = "/data/epione/user/huili/exp_coES/00023-MIMIC-auto25k-Greg-coGD/network-snapshot-006854.pkl"
        assert latent is not None
        # if latent is None:
            # latent = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/w_00023/"

        # Load network.
        print('Loading G network from "%s"...' % g_ckpt)
        with open(g_ckpt, 'rb') as f:
            G_network_dict = pickle.load(f)
        # Validate contents.
        if g_key is None:
            g_key='G_ema'
        assert isinstance(G_network_dict[g_key], torch.nn.Module)
        G = G_network_dict[g_key].eval().requires_grad_(False).to(device)

        extraction_latent(device, G, data_loader, latent)

    print('Time {:.3f} min'.format((time.time() - start_time) / 60))
    print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
