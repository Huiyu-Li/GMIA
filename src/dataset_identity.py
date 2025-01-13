import queue as Queue
import threading
import numpy as np
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

class MIMICR2_Ver(Dataset):
    r"""
     Iterating: two real-reconstructed image pairs
     For CallBackVerification during training
    """
    def __init__(self, 
                 csv_path, 
                 image_real_root='',
                 image_reco_root='',
                 img_channels = 1,
                 transform = [False, False], # [RandomHorizontalFlip, Rotation]
                 shuffle=False,
                 seed=123,
                 ):
        self.image_real_root = image_real_root
        self.image_reco_root = image_reco_root
        self.img_channels = img_channels
        self.transform = transform

        # load data from csv
        self.df = pd.read_csv(csv_path)
  
        # Test
        # self.df = self.df[:1000]

        self._num_pairs = len(self.df)
        
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_pairs))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
    
    def __len__(self):
        return self._num_pairs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname1 = f'p{str(row["subject_id_1"])[:2]}/p{str(row["subject_id_1"])}/s{str(row["study_id_1"])}/{row["dicom_id_1"]}.jpg'
        fname2 = f'p{str(row["subject_id_2"])[:2]}/p{str(row["subject_id_2"])}/s{str(row["study_id_2"])}/{row["dicom_id_2"]}.jpg'
        issame = row["issame"]
        image_path1 = f'{self.image_real_root}/{fname1}'
        image_path2 = f'{self.image_real_root}/{fname2}'

        if self.img_channels==3:
            image1 = Image.open(image_path1).convert('RGB') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('RGB') # PIL.Image.Image, [Columns, Rows]
        else:
            image1 = Image.open(image_path1).convert('L') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('L') # PIL.Image.Image, [Columns, Rows]

        if all(self.transform):
            image1_hflip = F.hflip(image1)
            image2_hflip = F.hflip(image2)
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip), F.to_tensor(image2_rotate)]
        elif self.transform[0]:
           image1_hflip = F.hflip(image1)
           image2_hflip = F.hflip(image2)
           image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip)] 
           image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip)]
        elif self.transform[1]:
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_rotate)]
        else:
            image1 = F.to_tensor(image1)
            image2 = F.to_tensor(image2)
        return image1, image2, issame
 
class MIMICI(Dataset):
    r"""
     Iterating: two anonymized image pairs
     For Linkability Inner Risk computing
    """
    def __init__(self, 
                 csv_path, 
                 image_root='',
                 img_channels = 1,
                 transform = [False, False], # [RandomHorizontalFlip, Rotation]
                 shuffle=False,
                 seed=123,
                 ):
        self.image_root = image_root
        self.img_channels = img_channels
        self.transform = transform

        # load data from csv
        self.df = pd.read_csv(csv_path)
  
        # Test
        # self.df = self.df[:1000]

        self._num_pairs = len(self.df)
        
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_pairs))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
    
    def __len__(self):
        return self._num_pairs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname1 = f'p{str(row["subject_id_1"])[:2]}/p{str(row["subject_id_1"])}/s{str(row["study_id_1"])}/{row["dicom_id_1"]}/X_A.jpg'
        fname2 = f'p{str(row["subject_id_2"])[:2]}/p{str(row["subject_id_2"])}/s{str(row["study_id_2"])}/{row["dicom_id_2"]}/X_A.jpg'
        issame = row["issame"]
        image_path1 = f'{self.image_root}/{fname1}'
        image_path2 = f'{self.image_root}/{fname2}'

        if self.img_channels==3:
            image1 = Image.open(image_path1).convert('RGB') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('RGB') # PIL.Image.Image, [Columns, Rows]
        else:
            image1 = Image.open(image_path1).convert('L') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('L') # PIL.Image.Image, [Columns, Rows]

        if all(self.transform):
            image1_hflip = F.hflip(image1)
            image2_hflip = F.hflip(image2)
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip), F.to_tensor(image2_rotate)]
        elif self.transform[0]:
           image1_hflip = F.hflip(image1)
           image2_hflip = F.hflip(image2)
           image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip)] 
           image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip)]
        elif self.transform[1]:
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_rotate)]
        else:
            image1 = F.to_tensor(image1)
            image2 = F.to_tensor(image2)
        return image1, image2, issame
    
class MIMICO(Dataset):
    r"""
     Iterating: real-anonymized image pairs 
     For Linkability Outer Risk computing
    """
    def __init__(self, 
                 csv_path, 
                 image_root='',
                 anony_root='',
                 mode = 'test',
                 img_channels = 1,
                 transform = [False, False], # [RandomHorizontalFlip, Rotation]
                 shuffle=False,
                 seed=123,
                 ):
        self.image_root = image_root
        self.anony_root = anony_root
        self.mode = mode
        self.img_channels = img_channels
        self.transform = transform

        # load data from csv
        self.df = pd.read_csv(csv_path)

        # read train-valid-test split
        if self.mode == "train":
            self.df = self.df[self.df['split']=='train']
        elif self.mode == "valid":
            self.df = self.df[self.df['split']=='validate']
        elif self.mode == "test":
            self.df = self.df[self.df['split']=='test']
        else:
            raise NotImplementedError(f"split {self.mode} is not implemented!")
        
        # Test
        # self.df = self.df[:1000]

        self._num_pairs = len(self.df)
        
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_pairs))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
    
    def __len__(self):
        return self._num_pairs
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname1 = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg'
        fname2 = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}/X_A.jpg'
        issame = 1 # 1:if only use same pairs
        image_path1 = f'{self.image_root}/{fname1}'
        image_path2 = f'{self.anony_root}/{fname2}'
    
        if self.img_channels==3:
            image1 = Image.open(image_path1).convert('RGB') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('RGB') # PIL.Image.Image, [Columns, Rows]
        else:
            image1 = Image.open(image_path1).convert('L') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('L') # PIL.Image.Image, [Columns, Rows]

        if all(self.transform):
            image1_hflip = F.hflip(image1)
            image2_hflip = F.hflip(image2)
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip), F.to_tensor(image2_rotate)]
        elif self.transform[0]:
           image1_hflip = F.hflip(image1)
           image2_hflip = F.hflip(image2)
           image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip)] 
           image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip)]
        elif self.transform[1]:
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_rotate)]
        else:
            image1 = F.to_tensor(image1)
            image2 = F.to_tensor(image2)
        return image1, image2, issame
    