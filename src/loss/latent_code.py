import os
import torch
import torch.nn as nn

class LatentCode(nn.Module):
    def __init__(self, Z_real):
        """Anonymization latent code class.
        Args:
            Z_real (torch.Tensor) : latent code Z of real image
        """
        super(LatentCode, self).__init__()
        
        # Define and initialise latent code parameters
        self.trainable_layers = nn.Parameter(data=Z_real, requires_grad=True) # Defult: requires_grad=True

    def do_optim(self, latent_root, fname):
        """Check whether optimization latent code has been saved."""
        n_samples = len(fname)
        if n_samples > 1: # For batchsize>1
            for i in range(n_samples):
                save_dir_Z = os.path.join(latent_root, fname[i][:-4], 'Z_op.pt')
                save_dir_A = os.path.join(latent_root, fname[i][:-4], 'X_A.jpg')
                if os.path.exists(save_dir_Z) and os.path.exists(save_dir_A):
                   continue
                else:  
                    return True
            return False
        else: # For batchsize=1
            save_dir_Z = os.path.join(latent_root, fname[0][:-4], 'Z_op.pt')
            save_dir_A = os.path.join(latent_root, fname[0][:-4], 'X_A.jpg')
            if os.path.exists(save_dir_Z) and os.path.exists(save_dir_A) :
                return False
            else:
                return True

    def save(self, latent_root, fname):
        """Save anonymization latent code."""
        Z = self.trainable_layers
        n_samples = Z.shape[0] # [NCL]
        
        if n_samples > 1: # For batchsize>1
            save_dir = []
            for i in range(n_samples):
                piece_dir = os.path.join(latent_root, fname[i][:-4])
                if not os.path.exists(piece_dir):
                    os.makedirs(piece_dir)
                torch.save(Z[i].detach().cpu(), os.path.join(piece_dir, 'Z_op.pt'))
                save_dir.append(piece_dir)

        else: # For batchsize=1
            save_dir = os.path.join(latent_root, fname[0][:-4])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(Z[0].detach().cpu(), os.path.join(save_dir, 'Z_op.pt')) # save in W+ space

        return save_dir, n_samples
    
    def forward(self):
        return self.trainable_layers
 