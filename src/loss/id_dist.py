'''
Loss to enlarge the distance between the 𝑍_𝑖𝑑^𝑅 and 𝑍_𝑖𝑑^𝐴
'''
import torch
from torch import nn
import torch.nn.functional as F

from src.network.iresnet import iresnet50, iresnet100

class IDLoss(nn.Module):
    def __init__(self, device, id_margin=0.0, network="r50", in_channel=1, embedding_size=512, id_ckpt=None):
        super(IDLoss, self).__init__()
        self.id_margin = id_margin
        self.target = torch.tensor([-1]).to(device)

        # Loading Identity Encoder 
        # More model choice refer to My_ArcFace foler
        if network == "r50":
            backbone = iresnet50(in_channel=in_channel, dropout=0.0, num_features=embedding_size)
        elif network == "r100":
            backbone = iresnet100(in_channel=in_channel, dropout=0.0, num_features=embedding_size)
        dict_checkpoint = torch.load(id_ckpt)
        # dict_keys(['epoch', 'global_step', 'state_dict_backbone', 'state_dict_softmax_fc', 'state_optimizer', 'state_lr_scheduler'])
        backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        del dict_checkpoint
        self.model_identity = backbone

        # self.model_identity = backbone.eval().requires_grad_(False).to(device)
        # self.model_identity = backbone.eval().to(device)
        # Frozen the params
        # for param in self.model_identity.parameters():
        #     param.requires_grad = False

    def forward(self, S_real, X_anony):
        S_anony = self.model_identity(X_anony) # [N,512]
        
        assert S_real.shape == S_anony.shape, 'Size mismatch!'
        # print(f'ID==> S_real: {S_real.shape}, S_anony: {S_anony.shape}')

        # n_samples = S_real.shape[0]
        # S_real_flat = S_real.view(n_samples, -1)
        # Z_anony_flat = Z_anony.view(n_samples, -1)
        # print(f'S_real: {S_real_flat.shape}, Z_anony: {Z_anony_flat.shape}')
  
        loss = F.cosine_embedding_loss(S_real, S_anony, target=self.target, 
                                       margin=self.id_margin, reduction='mean')
        # # Check
        # # Cosine distance
        # cosine_similarity = F.cosine_similarity(S_real, S_anony)
        # # l2 distance
        # S_real_normalized = F.normalize(S_real, p=2.0, dim=1, eps=1e-12)
        # S_anony_normalized = F.normalize(S_anony, p=2.0, dim=1, eps=1e-12)
        # diff = S_real_normalized - S_anony_normalized
        # dist = torch.sum(torch.square(diff), dim=1)
        # print(f'cosine: {cosine_similarity.item():.3f}, dist: {dist.item():.3f}, {dist.item()> 1.3580}') # threshold:1.3580

        return loss