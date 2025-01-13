# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import json
import pickle
import torch
from visdom import Visdom
import numpy as np
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

# Custom modules
import sys
sys.path.append('/home/huili/Projects/my_coES_stylegan2/')
import dnnlib
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix
from src.loss.latent_code import LatentCode
from src.loss.id_dist import IDLoss
from src.loss.ut_sim import UTLoss
#----------------------------------------------------------------------------

def vis_images(vis, images, fname, clamp=True):
    # print(f'real[ {real.min()}, {real.max()} ]') #[0,1]
    # print(f'decoded[ {predict.min()}, {predict.max()} ]') #[-0.9,1.1]

    # if args.normalize is True:
    #     # [-1.,1.] -> [0.,1.]
    #     reconstruction = (reconstruction + 1.) / 2.  
    if clamp:
        images = torch.clamp(images, min=0., max=1.)

    vis.images(images, win='image', opts=dict(title=f'real-recover-anonymize: {fname}', nrow=3))#(batch, 1, width, height)

    # if fname:
    #     fname_list = [f'{item} \n' for item in fname]
    #     vis.text(fname_list, win='fname', opts=dict(title='fname'))

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    vis_kwargs              = {},       # Options for visdom.
    dataset_kwargs          = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    G_ckpt                  = None,     # Pre-trained generator checkpoint
    W_opt_kwargs            = {},       # Options for latent code optimizer.
    id_loss_kwargs          = {},       # Options for identity loss function.
    ut_loss_kwargs          = {},       # Options for utility loss function.
    # w_plus                  = False,    # Optimize in the W+ space.
    lambda_id               = 1,        # Coefficient of id loss.
    lambda_ut               = 1,        # Coefficient of ut loss.
    # lambda_w                = 0,        # Coefficient of w loss.
    # w_cosine                = 0,        # Perform w cosine distance.
    # w_l2norm                = 0,        # Perform w l2norm distance.
    # w_il2norm               = 0,        # Perform w inverse l2norm distance.
    run_epochs              = 0,        # Number of running epochs.
    early_stop              = 0,        # Early stop threshold for identity loss.
    num_ws                  = 0,        # Number of intermediate latents to output.
    random_seed             = 0,        # Global random seed.
    batch_size              = 1,        # Total batch size for one training iteration.
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda')
    
    # Reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False              # Improves training speed. False for producibility
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.

    # Load training set.
    print('Loading training set...')
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    data_loader_kwargs = dnnlib.EasyDict(batch_size=batch_size, shuffle=False, num_workers=3, prefetch_factor=2, 
                                         collate_fn=None, pin_memory=True, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, **data_loader_kwargs)
    print('Num images: ', len(dataset))

    # Loss
    S_id_loss = IDLoss(device, **id_loss_kwargs).eval().requires_grad_(False).to(device)
    S_ut_loss = UTLoss(device, **ut_loss_kwargs).eval().requires_grad_(False).to(device)

    # Construct networks.
    # print('Constructing networks...')
    # G = dnnlib.util.construct_class_by_name(**G_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    # Resume from existing pickle.
    print(f'Resuming G from "{G_ckpt}"')  
    with open(G_ckpt, 'rb') as f:
        resume_data = pickle.load(f)
    # misc.copy_params_and_buffers(resume_data['G_ema'], G, require_all=False)
    G = resume_data['G_ema'].eval().requires_grad_(False).to(device)

    # Print network summary tables.
    ws_demo = torch.empty([1, num_ws, G.z_dim], device=device)
    misc.print_module_summary(G.synthesis, [ws_demo])

    # Initialize logs.
    print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_jsonl = None
    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')

    vis = Visdom(**vis_kwargs)
    # Main loop.
    for idx, data in enumerate(data_loader, 0):
    # for fname, image, Z_real, S_real_id, S_real_ut in data_loader:
        fname, image_real, Z_real, S_real_id, S_real_ut = data
        image_real = image_real.to(device, dtype=torch.float32)
        Z_real = Z_real.to(device, dtype=torch.float32)
        # if w_plus:
        #     Z_real = Z_real.unsqueeze(1).repeat([1, num_ws, 1])# Broadcast
        S_real_id = S_real_id.to(device, dtype=torch.float32)
        S_real_ut = S_real_ut.to(device, dtype=torch.float32)

        Z_real0 = Z_real.clone()
        '''
        # Check
        image_real0 = image_real.clone()
        S_real_id0 = S_real_id.clone()
        S_real_ut0 = S_real_ut.clone()
        '''

        # origin_Z_real = Z_real.clone() # Check Optimization
        # Build anonymization latent code
        latent_code = LatentCode(Z_real).to(device)

        # Check whether anonymization latent code has already been optimized -- if so, continue with the next one
        if not latent_code.do_optim(f'{run_dir}/anony', fname):
            print('latent_code already optimized!')
            continue
        
        # optimizer
        optimizer = dnnlib.util.construct_class_by_name(params=latent_code.parameters(), **W_opt_kwargs) # subclass of torch.optim.Optimizer

        # Set learning rate scheduler
        # lr_scheduler = MultiStepLR(optimizer=optimizer,
        #                            milestones=[int(m * args.epochs) for m in args.lr_milestones],
        #                            gamma=args.lr_gamma, verbose='True')
        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience)

        # Zero out gradients
        G.zero_grad()
        S_id_loss.zero_grad()
        S_ut_loss.zero_grad()
        # if lambda_id: 
        #     S_id_loss.zero_grad()
        # if lambda_ut:
        #     S_ut_loss.zero_grad()
        
        for epoch in range(run_epochs):
            # check_model_status(logger)

            # Clear gradients wrt parameters
            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)

            # Generate anonymized image
            # if w_plus:
            #     img_anony = G.synthesis(latent_code())
            # else:
            img_anony = G.synthesis(latent_code().unsqueeze(1).repeat([1, num_ws, 1]))

            # Calculate identity and utility losses
            id_loss = S_id_loss(S_real_id, img_anony)
            weighted_id_loss = lambda_id * id_loss
            ut_loss = S_ut_loss(S_real_ut, img_anony)
            weighted_ut_loss = lambda_ut * ut_loss
            loss = weighted_id_loss + weighted_ut_loss
            fields_check = f'epoch: {epoch}, id_loss:{id_loss:.5f}, ut_loss:{ut_loss:.5f}' 

            # loss = 0.
            # fields_check = f'epoch: {epoch}' 
            # if lambda_id:
            #     id_loss = S_id_loss(S_real_id, img_anony)
            #     weighted_id_loss = lambda_id * id_loss
            #     loss += weighted_id_loss
            #     fields_check += f', id_loss:{id_loss:.5f}'
            # if lambda_ut:
            #     ut_loss = S_ut_loss(S_real_ut, img_anony)
            #     weighted_ut_loss = lambda_ut * ut_loss
            #     loss += weighted_ut_loss
            #     fields_check += f', ut_loss:{ut_loss:.5f}'

            '''
            if lambda_w:
                # Normalize the input tensors
                # Z_real0_normalized = Z_real0 / (torch.norm(Z_real0, p=2)+1e-12)
                # Z_anony_normalized = latent_code() / (torch.norm(latent_code(), p=2)+1e-12)
                Z_real0_normalized = F.normalize(Z_real0, p=2.0, dim=-1, eps=1e-12)
                Z_anony_normalized = F.normalize(latent_code(), p=2.0, dim=-1, eps=1e-12)
                # print(f'norm:{torch.norm(Z_real0_normalized, p=2)}, {torch.norm(Z_anony_normalized, p=2)}')
                # Define target
                target=torch.tensor([-1]).to(device)

                if w_cosine:
                    w_loss = F.cosine_embedding_loss(Z_real0_normalized, Z_anony_normalized,
                                                     target, margin=-1., reduction='mean')
                elif w_l2norm:    
                    input = torch.norm(Z_real0_normalized - Z_anony_normalized, p=2)
                    w_loss = F.hinge_embedding_loss(input, target, margin=1.0, reduction='mean')
                elif w_il2norm:
                    input = torch.norm(Z_real0_normalized - Z_anony_normalized, p=2)
                    w_loss = 1 / (input + 1e-12)

                weighted_w_loss = lambda_w * w_loss
                loss += weighted_w_loss
                fields_check += f', w_loss:{w_loss:.5f}'
            '''

            # Check
            print(fields_check)
                
            # Back-propagation
            loss.backward()

            for param in latent_code.parameters():
                if param.grad is not None:
                    misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer.step()
            # lr_scheduler.step(loss) # Actually should use val_loss instead of train_loss

            # Collect statistics.
            stats_collector.update()
            stats_dict = stats_collector.as_dict()

            # Update logs.
            timestamp = time.time()
            if stats_jsonl is not None:
                fields = dict(stats_dict, timestamp=timestamp)
                stats_jsonl.write(json.dumps(fields) + '\n')
                stats_jsonl.flush()

            training_stats.report('Loss/id', id_loss)
            training_stats.report('Loss/ut', ut_loss)
            vis.line(Y=np.array([id_loss.item()]), X=np.array([epoch]), win='id', name='identity', 
                        update=None if epoch==0 else 'append', 
                        opts=dict(title='identity', showlegend=True))
            vis.line(Y=np.array([ut_loss.item()]), X=np.array([epoch]), win='ut', name='utility', 
                        update=None if epoch==0 else 'append', 
                        opts=dict(title='utility', showlegend=True))
            
            # if lambda_id:
            #     training_stats.report('Loss/id', id_loss)
            #     vis.line(Y=np.array([id_loss.item()]), X=np.array([epoch]), win='id', name='identity', 
            #             update=None if epoch==0 else 'append', 
            #             opts=dict(title='identity', showlegend=True))
            # if lambda_ut:
            #     training_stats.report('Loss/ut', ut_loss)
            #     vis.line(Y=np.array([ut_loss.item()]), X=np.array([epoch]), win='ut', name='utility', 
            #             update=None if epoch==0 else 'append', 
            #             opts=dict(title='utility', showlegend=True))
                
            # if lambda_w and epoch>1:
            #     training_stats.report('Loss/w', w_loss)
            #     vis.line(Y=np.array([w_loss.item()]), X=np.array([epoch]), win='w', name='w_reg', 
            #         update=None if epoch==2 else 'append', 
            #         opts=dict(title='w_reg', showlegend=True))
                
            # early stop
            if lambda_id and id_loss < early_stop:
                break
        
        # End epoch
        # mean_epoch_loss = np.mean(epoch_loss)
        fields1 = f'===>> Index {idx} | ID loss:{id_loss:.5f} | UT loss: {ut_loss:.5f}'

        # fields1 = f'===>> Index {idx}'
        # if lambda_id:
        #     fields1 += f' | ID loss:{id_loss:.5f}'
        # if lambda_ut:
        #     fields1 += f' | UT loss: {ut_loss:.5f}'
        # if lambda_w:
        #     fields1 += f' | W loss: {w_loss:.5f}'
        print(fields1)  
        
        # Store optimized anonymization latent codes
        save_dir, n_samples = latent_code.save(f'{run_dir}/anony', fname)
        # Generate and save anonymized image
        with torch.no_grad():
            '''
            # Check
            label = torch.zeros([1, G.c_dim], device=device)

            # print(f'img - img:{(image_real0-image_real).sum()}')
            # print(f'Z_real0 - Z_real:{(Z_real0 - Z_real).sum()}')
            # print(f'S_real_id0 - S_real_id:{(S_real_id0 - S_real_id).sum()}')
            # print(f'S_real_ut0 - S_real_ut:{(S_real_ut0 - S_real_ut).sum()}')

            Z_real1 = G.encoder(image_real, label)
            gen_img1 = G.synthesis(Z_real1, noise_mode='const')
            # if w_plus:
            #     gen_img0 = G.synthesis(Z_real0, noise_mode='const')
            # else:
            gen_img0 = G.synthesis(Z_real0.unsqueeze(1).repeat([1, num_ws, 1]), noise_mode='const')
            gen_img2 = G(image_real, label, noise_mode='const') 
            print(f'Z_real0 - Z_real1:{(Z_real0 - Z_real1).sum()}')
            print(f'gen_img0 - gen_img1:{(gen_img0-gen_img1).sum()}')
            print(f'gen_img0 - gen_img2:{(gen_img0-gen_img2).sum()}')
            print(f'gen_img1 - gen_img2:{(gen_img0-gen_img2).sum()}')

            # if w_plus:
            #     gen_img3 = G.synthesis(Z_real, noise_mode='const')
            #     img_anony_final = G.synthesis(latent_code(), noise_mode='const') # [N,1,256,256]
            # else:
            gen_img3 = G.synthesis(Z_real.unsqueeze(1).repeat([1, num_ws, 1]), noise_mode='const')
            img_anony_final = G.synthesis(latent_code().unsqueeze(1).repeat([1, num_ws, 1]), noise_mode='const') # [N,1,256,256]
            print(f'gen_img3 - img_anony_final:{(gen_img3-img_anony_final).sum()}')    
            print()
   
            S_real_id1 = S_id_loss.model_identity(image_real) # [N,512]
            print(f'S_real_id0 - S_real_id1:{(S_real_id0 - S_real_id1).sum()}')
            
            image_real_3 = image_real.clone().repeat(1,3,1,1) # [1, 1, 256, 256]
            image_real_3 = S_ut_loss._transforms_()(image_real_3)
            S_real_ut1 = S_ut_loss.model_utility.encoder(image_real_3) # [N,512]
            print(f'S_real_ut0 - S_real_ut1:{(S_real_ut0 - S_real_ut1).sum()}')               
            print()
            # End Check
            '''
            
            # if w_plus:
            #     gen_img0 = G.synthesis(Z_real0, noise_mode='const')
            #     img_anony_final = G.synthesis(latent_code(), noise_mode='const') # [N,1,256,256]
            # else:
            gen_img0 = G.synthesis(Z_real0.unsqueeze(1).repeat([1, num_ws, 1]), noise_mode='const')
            img_anony_final = G.synthesis(latent_code().unsqueeze(1).repeat([1, num_ws, 1]), noise_mode='const') # [N,1,256,256]
            if n_samples > 1: # For batchsize>1
                for i in range(n_samples):
                    save_image(img_anony_final[i], fp=os.path.join(save_dir[i], 'X_A.jpg'))
            else: # For batchsize=1  
                save_image(img_anony_final, fp=os.path.join(save_dir, 'X_A.jpg'))
                real_gen_img = torch.cat([image_real, gen_img0, img_anony_final]) # [1,1,256,256]>[1,256,256]
                print(f'real_img - img_anony_final:{(image_real-img_anony_final).mean():.5f}, gen_img0 - img_anony_final:{(gen_img0-img_anony_final).mean():.5f}')
                vis_images(vis, real_gen_img, fname=fname, clamp=True)
                # Check
                # save_image(real_gen_img, fp=os.path.join(save_dir, 'X_R.jpg'))
#----------------------------------------------------------------------------
