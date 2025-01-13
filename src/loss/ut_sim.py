'''
Loss to enhance the similarity between 𝑍_𝑢𝑡^𝑅 and 𝑍_𝑢𝑡^𝐴
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.network.densenet_utility import DenseNet121

class UTLoss(nn.Module):
    def __init__(self, device, input_nc=3, normalize=True, pretrained=False, # since DenseNet121(initialized from ImageNet27)
                activations='elu', last_activation=None, num_classes=4, ut_ckpt=None):
        
        super(UTLoss, self).__init__()
        self.input_nc=input_nc
        self.normalize = normalize

        # Loading Utiliyt Encoder
        model_utility = DenseNet121(num_in_features=input_nc, pretrained=pretrained, 
                            activations=activations, last_activation=last_activation, 
                            num_classes=num_classes)
        checkpoint_utility = torch.load(ut_ckpt)
        model_utility.load_state_dict(checkpoint_utility['model_state_dict'], strict=False)# `strict` False if warmstarting
        del checkpoint_utility
        self.model_utility = model_utility
        
        # self.model_utility = model_utility.eval().requires_grad_(False).to(device)
        # self.model_utility = model_utility.eval().to(device)
        # Frozen the params
        # for param in self.model_utility.parameters():
        #     param.requires_grad = False

    def _transforms_(self):            
        return transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

    def forward(self, S_real, X_anony):
        if self.input_nc==3: 
            X_anony = X_anony.repeat(1,3,1,1) # [1, 1, 256, 256]
        if self.normalize:
            X_anony = self._transforms_()(X_anony) 

        # print(f'==> X_anony:{X_anony.shape}')
        S_anony = self.model_utility.encoder(X_anony) # [N,512]

        assert S_real.shape == S_anony.shape, 'Size mismatch!'
        # print(f'UT==> S_real: {S_real.shape}, S_anony: {S_anony.shape}')

        # l1_err = F.smooth_l1_loss(S_real, S_anony, reduction='mean', beta=1.0)
        sq_err = F.mse_loss(S_real, S_anony, reduction='mean')
        # print(f'ut loss====>l1_err: {l1_err.item():.3f}, sq_err: {sq_err.item():.3f}')
        return sq_err
