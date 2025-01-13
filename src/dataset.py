import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

def exception_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class MIMIC(Dataset):
    r"""
     Iterating: image
    """
    def __init__(self, 
                 csv_path, 
                 image_root='',
                 mode='train',
                 img_channels = 1,
                 use_transform = True, # Defualt: True
                 normalize = False,
                 use_labels=False,

                 class_index=-1,
                 use_softlabel=False,
                 use_upsampling=False,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'], # class imbalance
                 train_cols=['Lung Opacity', 'Atelectasis', 'Pleural Effusion', 'Support Devices'],

                 shuffle=False,
                 seed=123,
                 
                 oui = [True, False, False]# use [Origin image, Utility label, Identity label]
                 ):
        self.image_root = image_root
        self.mode = mode
        self.img_channels=img_channels
        self.use_transform = use_transform
        self.normalize = normalize
        self.use_labels = use_labels
        if self.use_labels is False:
          self.label_dim = 0

        self.class_index = class_index
        #assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        self.train_cols = train_cols

        self.oui = oui

        # load data from csv
        self.df = pd.read_csv(csv_path)
        
        if self.oui[2]:
            self.num_identity = len(pd.unique(self.df["subject_id"]))

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

        if self.oui[1]:
            # upsample selected cols
            if use_upsampling:
                assert isinstance(upsampling_cols, list), 'Input should be list!'
                sampled_df_list = []
                for col in upsampling_cols:
                    print ('Upsampling %s...'%col)
                    sampled_df_list.append(self.df[self.df[col] == 1])
                self.df = pd.concat([self.df] + sampled_df_list, axis=0)   
            # impute missing values 
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']:
                    if use_softlabel:
                        y_hat = np.random.uniform(low=0.55, high=0.85, size=None)
                        self.df[col].replace(-1, y_hat, inplace=True)    
                    else:
                        self.df[col].replace(-1, 1, inplace=True)  
                else:
                    if use_softlabel:
                        y_hat = np.random.uniform(low=0.0, high=0.3, size=None)
                        self.df[col].replace(-1, y_hat, inplace=True) 
                    else:
                        self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)  

        self.max_size = len(self.df)
        
        # shuffle data
        if shuffle:
            data_index = list(range(self.max_size))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
    
    def __len__(self):
        return self.max_size
    
    def get_class_instance_num(self): # Based on TrainSet
        if self.mode == "train" and self.oui[1]:
            column_nums=self.df.loc[:,self.train_cols].apply(lambda x: x.sum(), axis=0) # check the sum for each column: df["col1"].sum()
            class_instance_nums = column_nums.values.tolist()
            return self.max_size, class_instance_nums
        else:
            raise NotImplementedError(f"{self.mode} is not implemented!")
    
    def _transforms_(self):       
        if self.mode == "train" and self.use_transform:
            transforms_list = [transforms.RandomHorizontalFlip(p=0.5),
                              transforms.RandomRotation(15),
                              # transforms.Resize(256),
                              # transforms.CenterCrop(256),
                              transforms.ToTensor()
                              ]
        else:
            transforms_list = [# transforms.Resize(256),
                              # transforms.CenterCrop(256),
                              transforms.ToTensor()
                              ]
      
        if self.normalize is True:
            transforms_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]
            # mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5] # len=1 for 1 channel image
            # mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
            
        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg'
        image_path = f'{self.image_root}/{fname}'
        
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # numpy.ndarray, uint8, [Rows, Columns]
        if self.img_channels==3:
          image = Image.open(image_path).convert('RGB') # PIL.Image.Image, [Columns, Rows]
        else:
            image = Image.open(image_path).convert('L') # PIL.Image.Image, [Columns, Rows]
        
        image = self._transforms_()(image) # torch.Tensor, torch.float32, [Rows, Columns]
        
        # Test
        # from torchvision.utils import save_image
        # save_image(image, f'./__pycache__/image{idx}.png')

        # if self.img_channels==3 and len(image.shape[0]) == 1:
        #     image = image.repeat(3,1,1)

        if self.oui[1]:
            if self.class_index != -1: # multi-class mode
                utility_label = torch.tensor([row[self.train_cols].values[self.class_index]], dtype=torch.float32)
            else:
                utility_label = torch.from_numpy(row[self.train_cols].values.astype(np.float32))

        if self.oui[2]:
            # Vector Label
            # identity_label = torch.zeros(self.num_identity, dtype=torch.float32)
            # identity_label[row['Identity']] = 1.
            # Arcface Label
            identity_label = torch.tensor(row['Identity'], dtype=torch.long)
        
        # Assume label info is None
        label = np.zeros([0], dtype=np.float32)
    
        if all(self.oui):
            return fname, image.clone(), utility_label.clone(), identity_label.clone()
        elif self.oui[0]:
            return fname, image.clone(), label.copy()
        elif self.oui[1]:
            return fname, image.clone(), utility_label.clone()
        elif self.oui[2]:
            return fname, image.clone(), identity_label.clone()
        else:
            raise NotImplementedError(f"{self.oui} is incorrect!")

class MIMICR2(Dataset):
    r"""
     Iterating: real-reconstructed image pairs
    """
    def __init__(self, 
                 csv_path, 
                 image_real_root='',
                 image_reco_root='',
                 mode='train',
                 img_channels = 1,
                 use_transform = True, # Defualt: True
                 normalize = False,
                 use_labels=False,

                 class_index=-1,
                 use_softlabel=False,
                 use_upsampling=False,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'], # class imbalance
                 train_cols=['Lung Opacity', 'Atelectasis', 'Pleural Effusion', 'Support Devices'],

                 shuffle=False,
                 seed=123,
                 
                 oui = [True, False, False]# use [Origin image, Utility label, Identity label]
                 ):
        self.image_real_root = image_real_root
        self.image_reco_root = image_reco_root
        self.mode = mode
        self.img_channels=img_channels
        self.use_transform = use_transform
        self.normalize = normalize
        self.use_labels = use_labels
        if self.use_labels is False:
          self.label_dim = 0

        self.class_index = class_index
        #assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        self.train_cols = train_cols

        self.oui = oui

        # load data from csv
        self.df = pd.read_csv(csv_path)
        
        if self.oui[2]:
            self.num_identity = len(pd.unique(self.df["subject_id"]))

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
        self.df = self.df[:100]

        if self.oui[1]:
            # upsample selected cols
            if use_upsampling:
                assert isinstance(upsampling_cols, list), 'Input should be list!'
                sampled_df_list = []
                for col in upsampling_cols:
                    print ('Upsampling %s...'%col)
                    sampled_df_list.append(self.df[self.df[col] == 1])
                self.df = pd.concat([self.df] + sampled_df_list, axis=0)   
            # impute missing values 
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']:
                    if use_softlabel:
                        y_hat = np.random.uniform(low=0.55, high=0.85, size=None)
                        self.df[col].replace(-1, y_hat, inplace=True)    
                    else:
                        self.df[col].replace(-1, 1, inplace=True)  
                else:
                    if use_softlabel:
                        y_hat = np.random.uniform(low=0.0, high=0.3, size=None)
                        self.df[col].replace(-1, y_hat, inplace=True) 
                    else:
                        self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)  

        self.max_size = len(self.df)
        
        # shuffle data
        if shuffle:
            data_index = list(range(self.max_size))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
    
    def __len__(self):
        return self.max_size
    
    def get_class_instance_num(self): # Based on TrainSet
        if self.mode == "train" and self.oui[1]:
            column_nums=self.df.loc[:,self.train_cols].apply(lambda x: x.sum(), axis=0) # check the sum for each column: df["col1"].sum()
            class_instance_nums = column_nums.values.tolist()
            return self.max_size, class_instance_nums
        else:
            raise NotImplementedError(f"{self.mode} is not implemented!")
    
    def _transforms_(self):       
        if self.mode == "train" and self.use_transform:
            transforms_list = [transforms.RandomHorizontalFlip(p=0.5),
                              transforms.RandomRotation(15),
                              # transforms.Resize(256),
                              # transforms.CenterCrop(256),
                              transforms.ToTensor()
                              ]
        else:
            transforms_list = [# transforms.Resize(256),
                              # transforms.CenterCrop(256),
                              transforms.ToTensor()
                              ]
      
        if self.normalize is True:
            transforms_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]
            # mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5] # len=1 for 1 channel image
            # mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
            
        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname_real = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg'
        fname_reco = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}/X_gen.png'
        fname_reco_ema = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}/X_gen_ema.png'

        image_path_real = f'{self.image_real_root}/{fname_real}'
        image_path_reco = f'{self.image_reco_root}/{fname_reco}'
        image_path_reco_ema = f'{self.image_reco_root}/{fname_reco_ema}'
        
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # numpy.ndarray, uint8, [Rows, Columns]
        if self.img_channels==3:
            image_real = Image.open(image_path_real).convert('RGB') # PIL.Image.Image, [Columns, Rows]
            image_reco = Image.open(image_path_reco).convert('RGB') 
            image_reco_ema = Image.open(image_path_reco_ema).convert('RGB') 
        else:
            image_real = Image.open(image_path_real).convert('L') # PIL.Image.Image, [Columns, Rows]
            image_reco = Image.open(image_path_reco).convert('L')
            image_reco_ema = Image.open(image_path_reco_ema).convert('L')

        image_real = self._transforms_()(image_real) # torch.Tensor, torch.float32, [Rows, Columns]
        image_reco = self._transforms_()(image_reco)
        image_reco_ema = self._transforms_()(image_reco_ema)

        # Test
        # from torchvision.utils import save_image
        # save_image(image, f'./__pycache__/image{idx}.png')

        # if self.img_channels==3 and len(image.shape[0]) == 1:
        #     image = image.repeat(3,1,1)

        if self.oui[1]:
            if self.class_index != -1: # multi-class mode
                utility_label = torch.tensor([row[self.train_cols].values[self.class_index]], dtype=torch.float32)
            else:
                utility_label = torch.from_numpy(row[self.train_cols].values.astype(np.float32))

        if self.oui[2]:
            # Vector Label
            # identity_label = torch.zeros(self.num_identity, dtype=torch.float32)
            # identity_label[row['Identity']] = 1.
            # Arcface Label
            identity_label = torch.tensor(row['Identity'], dtype=torch.long)
        
        # Assume label info is None
        label = np.zeros([0], dtype=np.float32)
    
        if all(self.oui):
            return fname_real, fname_reco, image_real.clone(), image_reco.clone(), image_reco_ema.clone(), utility_label.clone(), identity_label.clone()
        elif self.oui[0]:
            return fname_real, fname_reco, image_real.clone(), image_reco.clone(), image_reco_ema.clone(), label.copy()
        elif self.oui[1]:
            return image_real.clone(), image_reco.clone(), image_reco_ema.clone(), utility_label.clone()
        elif self.oui[2]:
            return image_real.clone(), image_reco.clone(), image_reco_ema.clone(), identity_label.clone()
        else:
            raise NotImplementedError(f"{self.oui} is incorrect!")

class MIMICA(Dataset):
    r"""
     Iterating: anonymized image
    """
    def __init__(self, 
                 csv_path, 
                 image_root='',
                 mode='train',
                 img_channels = 1,
                 use_transform = True, # Defualt: True
                 normalize = False,
                 use_labels=False,

                 class_index=-1,
                 use_softlabel=False,
                 use_upsampling=False,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'], # class imbalance
                 train_cols=['Lung Opacity', 'Atelectasis', 'Pleural Effusion', 'Support Devices'],

                 shuffle=False,
                 seed=123,
                 
                 oui = [True, False, False]# use [Origin image, Utility label, Identity label]
                 ):
        self.image_root = image_root
        self.mode = mode
        self.img_channels=img_channels
        self.use_transform = use_transform
        self.normalize = normalize
        self.use_labels = use_labels
        if self.use_labels is False:
          self.label_dim = 0

        self.class_index = class_index
        #assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        self.train_cols = train_cols

        self.oui = oui

        # load data from csv
        self.df = pd.read_csv(csv_path)
        
        if self.oui[2]:
            self.num_identity = len(pd.unique(self.df["subject_id"]))

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

        if self.oui[1]:
            # upsample selected cols
            if use_upsampling:
                assert isinstance(upsampling_cols, list), 'Input should be list!'
                sampled_df_list = []
                for col in upsampling_cols:
                    print ('Upsampling %s...'%col)
                    sampled_df_list.append(self.df[self.df[col] == 1])
                self.df = pd.concat([self.df] + sampled_df_list, axis=0)   
            # impute missing values 
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']:
                    if use_softlabel:
                        y_hat = np.random.uniform(low=0.55, high=0.85, size=None)
                        self.df[col].replace(-1, y_hat, inplace=True)    
                    else:
                        self.df[col].replace(-1, 1, inplace=True)  
                else:
                    if use_softlabel:
                        y_hat = np.random.uniform(low=0.0, high=0.3, size=None)
                        self.df[col].replace(-1, y_hat, inplace=True) 
                    else:
                        self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)  

        self.max_size = len(self.df)
        
        # shuffle data
        if shuffle:
            data_index = list(range(self.max_size))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
    
    def __len__(self):
        return self.max_size
    
    def get_class_instance_num(self): # Based on TrainSet
        if self.mode == "train" and self.oui[1]:
            column_nums=self.df.loc[:,self.train_cols].apply(lambda x: x.sum(), axis=0) # check the sum for each column: df["col1"].sum()
            class_instance_nums = column_nums.values.tolist()
            return self.max_size, class_instance_nums
        else:
            raise NotImplementedError(f"{self.mode} is not implemented!")
    
    def _transforms_(self):       
        if self.mode == "train" and self.use_transform:
            transforms_list = [transforms.RandomHorizontalFlip(p=0.5),
                              transforms.RandomRotation(15),
                              # transforms.Resize(256),
                              # transforms.CenterCrop(256),
                              transforms.ToTensor()
                              ]
        else:
            transforms_list = [# transforms.Resize(256),
                              # transforms.CenterCrop(256),
                              transforms.ToTensor()
                              ]
      
        if self.normalize is True:
            transforms_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]
            # mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5] # len=1 for 1 channel image
            # mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
            
        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}/X_A.jpg'

        image_path = f'{self.image_root}/{fname}'
        
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # numpy.ndarray, uint8, [Rows, Columns]
        if self.img_channels==3:
          image = Image.open(image_path).convert('RGB') # PIL.Image.Image, [Columns, Rows]
        else:
            image = Image.open(image_path).convert('L') # PIL.Image.Image, [Columns, Rows]
        
        image = self._transforms_()(image) # torch.Tensor, torch.float32, [Rows, Columns]
        
        # Test
        # from torchvision.utils import save_image
        # save_image(image, f'./__pycache__/image{idx}.png')

        # if self.img_channels==3 and len(image.shape[0]) == 1:
        #     image = image.repeat(3,1,1)

        if self.oui[1]:
            if self.class_index != -1: # multi-class mode
                utility_label = torch.tensor([row[self.train_cols].values[self.class_index]], dtype=torch.float32)
            else:
                utility_label = torch.from_numpy(row[self.train_cols].values.astype(np.float32))

        if self.oui[2]:
            # Vector Label
            # identity_label = torch.zeros(self.num_identity, dtype=torch.float32)
            # identity_label[row['Identity']] = 1.
            # Arcface Label
            identity_label = torch.tensor(row['Identity'], dtype=torch.long)
        
        # Assume label info is None
        label = np.zeros([0], dtype=np.float32)
    
        if all(self.oui):
            return fname, image.clone(), utility_label.clone(), identity_label.clone()
        elif self.oui[0]:
            return fname, image.clone(), label.copy()
        elif self.oui[1]:
            return fname, image.clone(), utility_label.clone()
        elif self.oui[2]:
            return fname, image.clone(), identity_label.clone()
        else:
            raise NotImplementedError(f"{self.oui} is incorrect!")
         
class MIMIC_Latent(Dataset):
    r"""
     Iterating: w latent, id and ut semantics
    """
    def __init__(self, 
                 csv_path, 
                 image_root='',
                 semantic_root = '',
                 latent_root = '',
                 mode='train',
                 start=None,
                 end=None,

                 img_channels = 1,
                 use_transform = False, # Defult: False
                 normalize = False,

                 shuffle=False, # Defult: False
                 seed=123,

                 load_z=False
                 ):
        self._name = 'MIMIC_Latent'
        self.image_root = image_root
        self.semantic_root = semantic_root
        self.latent_root = latent_root
        self.mode = mode
        self.img_channels=img_channels
        self.use_transform = use_transform
        self.normalize = normalize
        self.load_z = load_z

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
        # self.df = self.df[-2880:]
        if start is not None and end is not None:
            self.df = self.df[start:end]
            
        self.max_size = len(self.df)

        # shuffle data
        if shuffle:
            data_index = list(range(self.max_size))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
    
    def __len__(self):
        return self.max_size
    
    def _transforms_(self):       
        if self.mode == "train" and self.use_transform:
            transforms_list = [transforms.RandomHorizontalFlip(p=0.5),
                              transforms.RandomRotation(15),
                              # transforms.Resize(256),
                              # transforms.CenterCrop(256),
                              transforms.ToTensor()
                              ]
        else:
            transforms_list = [# transforms.Resize(256),
                              # transforms.CenterCrop(256),
                              transforms.ToTensor()
                              ]
      
        if self.normalize is True:
            transforms_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]
            # mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5] # len=1 for 1 channel image
            # mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
            
        return transforms.Compose(transforms_list)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg'
        image_path = f'{self.image_root}/{fname}'
        latent_path = f'{self.latent_root}/{fname[:-4]}'
        semantic_path = f'{self.semantic_root}/{fname[:-4]}'
        
        # Read Image
        image = Image.open(image_path).convert('L') # PIL.Image.Image, [Columns, Rows]
        image = self._transforms_()(image) # torch.Tensor, torch.float32, [Rows, Columns]

        # Read Latents
        S_id = torch.load(os.path.join(semantic_path, 'S_id.pt'))
        S_ut = torch.load(os.path.join(semantic_path, 'S_ut.pt'))
        # assert S_id.shape[0]==512 and S_ut.shape[0]==512, 'Shape should be [512]'
        
        # print(f'==>image:{image.min(), image.max()}')
        if self.load_z:
            Z_ge = torch.load(os.path.join(latent_path, 'W_ge.pt'))
            return fname, image, Z_ge, S_id, S_ut    
        else:
            return fname, image, S_id, S_ut
