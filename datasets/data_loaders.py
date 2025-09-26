import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from utils.utils import seed_everything

from scipy import signal
from scipy.signal import resample
import yaml

def get_dataloaders(data_name, train_val_test, resampling_rate, 
                    batch_size, num_workers=8,
                    signal_transform=None, random_seed=5):
    seed_everything(random_seed)
    
    if train_val_test == 'train':
        shuffle = True
    else:
        shuffle = False
        
    with open("./configs/dataset_configs.yaml", "r") as ymlfile:
        data_config = yaml.safe_load(ymlfile)
        
        
    if data_name == 'TUAB':
        t_dataset = TUABloader(data_dir= data_config['TUAB']['data_dir'],
                                train_val_test=train_val_test,
                                resampling_rate=resampling_rate,
                                signal_transform=signal_transform)
        t_loader = torch.utils.data.DataLoader(t_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle, 
                                                num_workers=num_workers)
        
    elif data_name == 'TUEV':
        t_dataset = TUEVloader(data_dir= data_config['TUEV']['data_dir'],
                                train_val_test=train_val_test,
                                resampling_rate=resampling_rate,
                                signal_transform=signal_transform)
        t_loader = torch.utils.data.DataLoader(t_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle, 
                                                num_workers=num_workers)
        
    elif data_name == 'CHBMIT':
        t_dataset = CHBMITloader(data_dir= data_config['CHBMIT']['data_dir'],
                                train_val_test=train_val_test,
                                resampling_rate=resampling_rate,
                                signal_transform=signal_transform)
        t_loader = torch.utils.data.DataLoader(t_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle, 
                                                num_workers=num_workers)
        
    elif data_name == 'EarEEG':
        t_dataset = EarEEGloader(data_dir= data_config['EarEEG']['data_dir'],
                                train_val_test=train_val_test,
                                resampling_rate=resampling_rate,
                                signal_transform=signal_transform)
        t_loader = torch.utils.data.DataLoader(t_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle, 
                                                num_workers=num_workers)
    return t_loader


# For Pretraining
def get_pretrain_dataloaders(resampling_rate, 
                    batch_size, num_workers=8,
                    signal_transform=None, 
                    dataset_list = ['TUAB','TUEV','CHBMIT','IIIC'],
                    random_seed=5):
    seed_everything(random_seed)
    
    t_dataset = UnsupervisedPretrainingloader(resampling_rate=resampling_rate,
                                              signal_transform=signal_transform,
                                              dataset_list=dataset_list)
    
    t_loader = torch.utils.data.DataLoader(t_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=num_workers,
                                            # pin_memory=True, persistent_workers=True,
                                            collate_fn=collate_fn_unsupervised_pretrain)
    
    return t_loader



class TUEVloader(Dataset):
    def __init__(self, data_dir, train_val_test, resampling_rate = 256, signal_transform=None):
        
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.default_sampling_rate = 256
        self.signal_len = 5
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        
        
        if train_val_test != 'test':
            self.data_files = os.listdir(os.path.join(data_dir,'processed_train',f'processed_{train_val_test}_split'))
        else:
            self.data_files = os.listdir(os.path.join(data_dir,'processed_eval'))
            
        print("Number of recordings from TUEV: ", len(self.data_files))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx): 
        if self.train_val_test != 'test':
            signal_data = pickle.load(open(os.path.join(self.data_dir,'processed_train',f'processed_{self.train_val_test}_split',self.data_files[idx]), 'rb'))
        else:
            signal_data = pickle.load(open(os.path.join(self.data_dir,'processed_eval',self.data_files[idx]), 'rb'))
        
        X = np.array(signal_data["signal"])
        labels = int(signal_data["label"][0] - 1)
        
        # resample
        if self.default_sampling_rate != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        
        X = torch.FloatTensor(X)
        
        return X, labels
    



class TUABloader(Dataset):
    def __init__(self, data_dir, train_val_test, resampling_rate = 256, signal_transform=None):
        
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.default_sampling_rate = 256
        self.signal_len = 10
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        
        self.data_files = os.listdir(os.path.join(data_dir,train_val_test))
        print("Number of recordings from TUAB: ", len(self.data_files))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx): 
        signal_data = pickle.load(open(os.path.join(self.data_dir,self.train_val_test,self.data_files[idx]), 'rb'))
        X = np.array(signal_data["X"])
        labels = signal_data["y"]
        
        # resample
        if self.default_sampling_rate != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        X = torch.FloatTensor(X)
        
        return X, labels
    
    

        
class CHBMITloader(Dataset):
    def __init__(self, data_dir, train_val_test, resampling_rate = 256, signal_transform=None):
        
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.default_sampling_rate = 256
        self.signal_len = 10
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        
        self.data_files = os.listdir(os.path.join(data_dir,train_val_test))
        print("Number of recordings from CHBMIT: ", len(self.data_files))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx): 
        signal_data = pickle.load(open(os.path.join(self.data_dir,self.train_val_test,self.data_files[idx]), 'rb'))
        X = np.array(signal_data["X"])
        labels = signal_data["y"]
        
        # resample
        if self.default_sampling_rate != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        X = torch.FloatTensor(X)
        
        return X, labels
    
class EarEEGloader(Dataset):
    def __init__(self, data_dir, train_val_test, resampling_rate = 256, signal_transform=None):
        
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.default_sampling_rate = 250
        self.signal_len = 30
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        
        self.data_files = os.listdir(os.path.join(data_dir,train_val_test))
        print("Number of recordings from EarEEG: ", len(self.data_files))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx): 
        signal_data = pickle.load(open(os.path.join(self.data_dir,self.train_val_test,self.data_files[idx]), 'rb'))
        X = np.array(signal_data["X"])
        X = X[:-1,:]
        labels = int(signal_data["label"])
        
        # resample
        if self.default_sampling_rate != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        
        X = torch.FloatTensor(X)
        
        return X, labels



# For Unsupervised Pretraining Loader
class UnsupervisedPretrainingloader(Dataset):
    def __init__(self, resampling_rate = 256, signal_transform=None, dataset_list = ['TUAB','TUEV','CHBMIT','IIIC']):
        
        
        with open("./configs/dataset_configs.yaml", "r") as ymlfile:
            self.data_config = yaml.safe_load(ymlfile)

        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        self.dataset_list = dataset_list
        
        self.tuab_data_files = []
        self.tuev_data_files = []
        self.chbmit_data_files = []
        self.iiic_data_files = []
        
        if 'TUAB' in self.dataset_list:
            # TUAB
            self.tuab_root = os.path.join(self.data_config['TUAB']['data_dir'],'train')
            self.tuab_data_files = os.listdir(self.tuab_root)
            print("Number of recordings from TUAB: ", len(self.tuab_data_files))
            if len(self.tuab_data_files) == 0:
                raise ValueError("No recordings found in TUAB")
        
        if 'TUEV' in self.dataset_list:
            # TUEV
            self.tuev_root = os.path.join(self.data_config['TUEV']['data_dir'],'processed_train',f'processed_train_split')
            self.tuev_data_files = os.listdir(self.tuev_root)
            print("Number of recordings from TUEV: ", len(self.tuev_data_files))
            if len(self.tuev_data_files) == 0:
                raise ValueError("No recordings found in TUEV")
            
        if 'CHBMIT' in self.dataset_list:
            # CHBMIT
            self.chbmit_root = os.path.join(self.data_config['CHBMIT']['data_dir'],'train')
            self.chbmit_data_files = os.listdir(self.chbmit_root)
            print("Number of recordings from CHBMIT: ", len(self.chbmit_data_files))
            if len(self.chbmit_data_files) == 0:
                raise ValueError("No recordings found in CHBMIT")
            
        if 'IIIC' in self.dataset_list:
            #IIIC
            self.iiic_root = os.path.join(self.data_config['IIIC']['data_dir'],'train')
            self.iiic_data_files = os.listdir(self.iiic_root)
            print("Number of recordings from IIIC: ", len(self.iiic_data_files))
            if len(self.iiic_data_files) == 0:
                raise ValueError("No recordings found in IIIC")
            
        print("Number of recordings from all datasets: ", len(self.tuab_data_files) + len(self.tuev_data_files) + len(self.chbmit_data_files) + len(self.iiic_data_files))
        
        
    def __len__(self):
        return len(self.tuab_data_files) + len(self.tuev_data_files) + len(self.chbmit_data_files) + len(self.iiic_data_files)
     
        
    def tuab_load(self,idx):
        signal_data = pickle.load(open(os.path.join(self.tuab_root,self.tuab_data_files[idx]), 'rb'))
        X = np.array(signal_data["X"])
        labels = signal_data["y"]
        
        # resample
        if self.data_config['TUAB']['sampling_rate'] != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.data_config['TUAB']['signal_length'] , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
            
        X = torch.FloatTensor(X)
        
        return X, labels, 0
    
    def tuev_load(self,idx):
        signal_data = pickle.load(open(os.path.join(self.tuev_root,self.tuev_data_files[idx]), 'rb'))
        
        X = np.array(signal_data["signal"])
        labels = int(signal_data["label"][0] - 1)
        
        # resample
        if self.data_config['TUEV']['sampling_rate'] != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.data_config['TUEV']['signal_length'] , axis=-1)
            
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        X = torch.FloatTensor(X)
        
        return X, labels, 1
    
    def chbmit_load(self,idx):
        signal_data = pickle.load(open(os.path.join(self.chbmit_root,self.chbmit_data_files[idx]), 'rb'))
        X = np.array(signal_data["X"])
        labels = signal_data["y"]
        
        # resample
        if self.data_config['CHBMIT']['sampling_rate'] != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.data_config['CHBMIT']['signal_length'] , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        X = torch.FloatTensor(X)
        
        return X, labels, 2
    
    def iiic_load(self,idx):
        signal_data = pickle.load(open(os.path.join(self.iiic_root,self.iiic_data_files[idx]), 'rb'))
        X = np.array(signal_data["signal"])
        labels = signal_data["label"]
        
        # resample
        if self.data_config['IIIC']['sampling_rate'] != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.data_config['IIIC']['signal_length'] , axis=-1)
            
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        X = torch.FloatTensor(X)
        
        return X, labels, 3
            
    

    def __getitem__(self, index):
        if 'TUAB' in self.dataset_list:
            if index < len(self.tuab_data_files):
                return self.tuab_load(index) 
        if 'TUEV' in self.dataset_list:
            if index < len(self.tuab_data_files) + len(self.tuev_data_files):
                index = index - len(self.tuab_data_files)
                return self.tuev_load(index)
        if 'CHBMIT' in self.dataset_list:
            if index < len(self.tuab_data_files) + len(self.tuev_data_files) + len(self.chbmit_data_files):
                index = index - len(self.tuab_data_files) - len(self.tuev_data_files)
                return self.chbmit_load(index)
        if 'IIIC' in self.dataset_list:
            if index < len(self.tuab_data_files) + len(self.tuev_data_files) + len(self.chbmit_data_files) + len(self.iiic_data_files):
                index = index - len(self.tuab_data_files) - len(self.tuev_data_files) - len(self.chbmit_data_files)
                return self.iiic_load(index)
        else:
            raise ValueError("Index out of range")
        
        
def collate_fn_unsupervised_pretrain(batch):
    tuab_samples, tuev_samples, chbmit_samples, iiic_samples = [], [], [], []
    
    for samples, labels, dataset_idx in batch:
        if dataset_idx == 0:
            tuab_samples.append(samples)
        elif dataset_idx == 1:
            tuev_samples.append(samples)
        elif dataset_idx == 2:
            chbmit_samples.append(samples)
        elif dataset_idx == 3:
            iiic_samples.append(samples)
        else:
            raise ValueError("Dataset index out of range")
            
    if len(tuab_samples) > 0:
        tuab_samples = torch.stack(tuab_samples)
    if len(tuev_samples) > 0:
        tuev_samples = torch.stack(tuev_samples)
    if len(chbmit_samples) > 0:
        chbmit_samples = torch.stack(chbmit_samples)
    if len(iiic_samples) > 0:
        iiic_samples = torch.stack(iiic_samples)
        
    tuab_samples = torch.tensor(tuab_samples)
    tuev_samples = torch.tensor(tuev_samples)
    chbmit_samples = torch.tensor(chbmit_samples)
    iiic_samples = torch.tensor(iiic_samples)
        
    return (tuab_samples, tuev_samples, chbmit_samples, iiic_samples)
