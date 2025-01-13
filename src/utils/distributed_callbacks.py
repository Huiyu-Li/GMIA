import sklearn
import numpy as np
import torch
# Custom modules
from torch_utils import training_stats
from src.metrics.metrics_identity import get_threshold

class CallBackVerification(object):
    def __init__(self, valid_loader, num_pairs, batch_size, transform, writer=None):
        self.highest_f1: float = 0.0
        self.highest_thr: float = 0.0
        self.best_f1_list = []
        self.best_thr_list = []
        
        # self.thresholds = np.arange(0, 4, 0.01) # 400
        self.thresholds = np.arange(0, 4, 0.001) # 4000
        self.valid_loader = valid_loader
        self.batch_size = batch_size
        self.num_pairs = num_pairs
        self.transform = transform
        self.writer = writer

    def ver_test(self, backbone: torch.nn.Module, epoch):
      issame_arr = np.zeros((self.num_pairs))
      dist_arr = np.zeros((self.num_pairs))
      left=0
      device = torch.device("cuda")
      with torch.no_grad():
          for idx, (img1, img2, issame) in enumerate(self.valid_loader):
              right = left+ self.batch_size
              if right>self.num_pairs:
                  print(f'right:{right}')
                  right = self.num_pairs
              issame_arr[left:right] = issame
              if all(self.transform):
                  image1 = img1[0].to(device); image2 = img2[0].to(device)
                  hflip1 = img1[1].to(device); hflip2 = img2[1].to(device)
                  rotate1 = img1[2].to(device); rotate2 = img2[2].to(device)
                  
                  embeddings_image1 = backbone(image1); embeddings_image2 = backbone(image2) 
                  embeddings_hflip1 = backbone(hflip1); embeddings_hflip2 = backbone(hflip2)
                  embeddings_rotate1 = backbone(rotate1); embeddings_rotate2 = backbone(rotate2)
                  # Check divice
                  # print(f'image1:{image1.device}, backbone:{backbone.device}, embeddings_image1:{embeddings_image1.device}')

                  embeddings1 = embeddings_image1+embeddings_hflip1+embeddings_rotate1
                  embeddings2 = embeddings_image2+embeddings_hflip2+embeddings_rotate2

              elif self.transform[0]:
                  image1 = img1[0].to(device); image2 = img2[0].to(device)
                  hflip1 = img1[1].to(device); hflip2 = img2[1].to(device)
                  embeddings_image1 = backbone(image1); embeddings_image2 = backbone(image2) 
                  embeddings_hflip1 = backbone(hflip1); embeddings_hflip2 = backbone(hflip2)

                  embeddings1 = embeddings_image1+embeddings_hflip1
                  embeddings2 = embeddings_image2+embeddings_hflip2

              elif self.transform[1]:
                  image1 = img1[0].to(device); image2 = img2[0].to(device)
                  rotate1 = img1[1].to(device); rotate2 = img2[1].to(device)

                  embeddings_image1 = backbone(image1); embeddings_image2 = backbone(image2)
                  embeddings_rotate1 = backbone(rotate1); embeddings_rotate2 = backbone(rotate2)
                  
                  embeddings1 = embeddings_image1+embeddings_rotate1
                  embeddings2 = embeddings_image2+embeddings_rotate2

              else:
                  image1 = img1.to(device); image2 = img2.to(device)
                  # image1 = img1; image2 = img2
                  embeddings1 = backbone(image1); embeddings2 = backbone(image2)

              # Scale input vectors individually to unit norm (vector length).
              embeddings1 = sklearn.preprocessing.normalize(embeddings1.detach().cpu().numpy()) # (N, 512)
              embeddings2 = sklearn.preprocessing.normalize(embeddings2.detach().cpu().numpy()) 
              # Check Size
              assert (embeddings1.shape[0] == embeddings2.shape[0])
              assert (embeddings1.shape[1] == embeddings2.shape[1])
              diff = np.subtract(embeddings1, embeddings2) # (N, 512)
              dist = np.sum(np.square(diff), 1) # (N)
              # Check Size np.sqrt(np.sum(embeddings1*embeddings1))
              dist_arr[left:right] = dist

              left = right

      # Calculate evaluation metrics
      best_threshold, best_f1_score, precision, recall, acc = get_threshold(
                                    self.thresholds, dist_arr, issame_arr)

      self.best_f1_list.append(best_f1_score)
      self.best_thr_list.append(best_threshold)
      print(f'F1 Score-Best: {best_f1_score:.3f}, Thr-Best:{best_threshold:.4f}')
      if best_f1_score > self.highest_f1:
          self.highest_f1 = best_f1_score
          self.highest_thr = best_threshold
          print(f'F1 Score-Highest: {self.highest_f1:.3f}, Thr-Highest: {self.highest_thr:.4f}')
      
      training_stats.report('Valid/best_threshold', best_threshold)
      training_stats.report('Valid/best_f1_score', best_f1_score)
      training_stats.report('Valid/precision', precision)
      training_stats.report('Valid/recall', recall)
      training_stats.report('Valid/acc', acc)
    #   if self.writer is not None:
    #     self.writer.add_scalar(tag='Valid/best_threshold', scalar_value=best_threshold, global_step=epoch)
    #     self.writer.add_scalar(tag='Valid/best_f1_score', scalar_value=best_f1_score, global_step=epoch)
    #     self.writer.add_scalar(tag='Valid/precision', scalar_value=precision, global_step=epoch)
    #     self.writer.add_scalar(tag='Valid/recall', scalar_value=recall, global_step=epoch)
    #     self.writer.add_scalar(tag='Valid/acc', scalar_value=acc, global_step=epoch)

    def __call__(self, backbone: torch.nn.Module, epoch):
        backbone.eval()
        self.ver_test(backbone, epoch)
        backbone.train()