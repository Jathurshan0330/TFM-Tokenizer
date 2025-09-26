import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from einops import rearrange,repeat
from timm.loss import LabelSmoothingCrossEntropy


import lightning as pl
from lightning.pytorch.loggers import NeptuneLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from utils.utils import seed_everything, cosine_scheduler, get_stft_torch
from utils.utils import   BCE, focal_loss, get_metrics
from datasets.data_loaders import get_dataloaders

from models.tfm_token import get_tfm_token_classifier_64x4
from models.tfm_token import get_tfm_tokenizer_2x2x8

class Pl_tfm_tokenizer_inference(pl.LightningModule):
    def __init__(self,args,training_params,save_path,niter_per_ep,dataset_params):
        super().__init__()
        self.args = args
        self.training_params = training_params
        self.save_path = save_path
        self.niter_per_ep = niter_per_ep
        self.dataset_params = dataset_params
        
        if self.dataset_params['classification_task']=='binary':
            self.metrics = ['accuracy','balanced_accuracy', 'roc_auc','pr_auc']
        else:
            self.metrics = ['accuracy','balanced_accuracy','cohen_kappa','f1_weighted']
            
        
        print(f'Loading TFM-TOKEN2 2x2x8 tokenizer model from: {args.vqvae_pretrained_path}')
        self.vqvae = get_tfm_tokenizer_2x2x8(code_book_size=args.code_book_size, emb_size=args.emb_size)
        self.vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path,map_location=self.device),strict=False)
        self.vqvae.to(self.device)
        self.vqvae.eval()

        print(f'Loading TFM-TOKEN classifier model: 64x4')
        self.tfm_token = get_tfm_token_classifier_64x4(n_classes=self.dataset_params['num_classes'],code_book_size=args.code_book_size, emb_size=args.emb_size)
        self.tfm_token.load_state_dict(torch.load(args.finetuned_path,map_location=self.device))
        self.tfm_token.to(self.device)
        self.tfm_token.eval()
        
        trainable_parameters = sum(p.numel() for p in self.tfm_token.parameters() if p.requires_grad)
        fixed_parameters = sum(p.numel() for p in self.tfm_token.parameters() if not p.requires_grad)
        with open(os.path.join(self.save_path,'tfm_encoder_model.txt'),'w') as f:
            f.write('Number of trainable parameters: \n')
            f.write(str(trainable_parameters))
            f.write('\n')
            f.write('Number of fixed parameters: \n')
            f.write(str(fixed_parameters))
            f.write('\n')
            f.write(str(self.tfm_token))
            f.close()
            
            
        self.val_step_outputs = []
        self.test_step_outputs = []
        
    
    def forward(self,x):
        x_temporal = x 
        B,C,T = x_temporal.shape
        # apply STFT
        x = get_stft_torch(x_temporal, resampling_rate = self.args.resampling_rate)
        x = rearrange(x,'B C F T -> (B C) F T').to(x_temporal.device)
        x_temporal = rearrange(x_temporal,'B C T -> (B C) T')

        with torch.no_grad():
            _,x_tokens,_ = self.vqvae.tokenize(x,x_temporal)
            x_tokens = rearrange(x_tokens,'(B C) T -> B C T', C=C)
            
        pred = self.tfm_token(x_tokens,num_ch = C)
        return pred
    
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        with torch.no_grad():
            pred = self(x)
            if self.dataset_params['classification_task']=='binary':
                pred = torch.sigmoid(pred)
                
            loss = 0
            step_pred = pred.cpu().numpy()
            step_y = y.cpu().numpy()
            self.test_step_outputs.append([step_pred,step_y])
        return loss
    
    def on_test_epoch_end(self):
        test_pred = np.concatenate([i[0] for i in self.test_step_outputs])
        test_y = np.concatenate([i[1] for i in self.test_step_outputs])
        
        if self.dataset_params['classification_task']=='binary':
            is_binary = True
        else:
            is_binary = False
            
        results = get_metrics(np.array(test_pred), np.array(test_y), self.metrics, is_binary)
        
        for i in range(len(self.metrics)):
            if i == 2:
                self.log(f'test_{self.metrics[i]}',results[self.metrics[i]],prog_bar=True,sync_dist=True)
            else:
                self.log(f'test_{self.metrics[i]}',results[self.metrics[i]],prog_bar=False,sync_dist=True)
        
        # save the results as csv
        results_df = pd.DataFrame(results,index = [0])
        results_df.to_csv(os.path.join(self.save_path,f'test_results_random_seed_{self.args.random_seed}_best.csv'),index=False)
        
        np.save(os.path.join(self.save_path,f'test_preds_random_seed_{self.args.random_seed}_best.npy'),test_pred)
        np.save(os.path.join(self.save_path,f'test_y_random_seed_{self.args.random_seed}_best.npy'),test_y)
        
        
        self.test_step_outputs = []
        
        return results
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='pretrain', help='Dataset name')
    parser.add_argument('--resampling_rate', type=int, default=200, help='Resampling rate')
    parser.add_argument('--gpu',type=str, default=None, help='GPU to use')
    
    parser.add_argument('--vqvae_pretrained_path', type=str, default=None, help='Path to the pretrained VQ-VAE model')
    parser.add_argument('--finetuned_path', type=str, default=None, help='Path to the pretrained model')
    parser.add_argument('--code_book_size', type=int, default=8192, help='Code book size')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--is_neptune', type=bool, default=False, help='Log to neptune')
    
    
    parser.add_argument('--random_seed', type=int, default=5, help='Random seed')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
    parser.add_argument('--test_only', action='store_true', help='Test only mode')
   
    args = parser.parse_args()
    
     # read the configuration file
    with open("./configs/tfm_tokenizer_training_configs.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
        
    # training parameters
    training_params = config['finetuning']
    
    
    # read the dataset configuration file
    with open("./configs/dataset_configs.yaml", "r") as ymlfile:
        dataset_config = yaml.safe_load(ymlfile)
    # dataset parameters
    dataset_params = dataset_config[args.dataset_name]
    
    experiment_name = f'TFM_TOKEN_Inference_{args.dataset_name}'
    if args.save_path is not None:
        training_params['experiment_path'] = args.save_path
    save_path = os.path.join(training_params['experiment_path'],experiment_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    #save the arguments and configuration in a text file
    with open(os.path.join(save_path,'args_config.txt'),'w') as f:
        f.write('Experiment name: ' + experiment_name + '\n')
        f.write('Arguments:\n')
        f.write(str(args))
        f.write('\n----------------------\n')
        f.write('Training parameters:\n')
        f.write(str(training_params))
        f.write('\n----------------------\n')
        f.write('Dataset parameters:\n')
        f.write(str(dataset_params))
        f.close()
        
    
    # Create the dataloaders
    seed_everything(args.random_seed)
        
    test_loader = get_dataloaders(data_name= args.dataset_name,
                                   train_val_test='test',
                                   resampling_rate=args.resampling_rate,
                                   batch_size=training_params['batch_size'], 
                                   num_workers=8,
                                   signal_transform = None,
                                   random_seed=args.random_seed)
    
    if args.is_neptune:
        # initializing logger
        logger = NeptuneLogger(
            api_key="[API_KEY]",  # replace with your own
            project="[PROJECT_NAME]",
            tags=[experiment_name],
            log_model_checkpoints=True,  # save checkpoints
        )
    else: 
        logger = CSVLogger(save_dir = save_path,name = 'logs')
    
    
    
    pl_finetune = Pl_tfm_tokenizer_inference(args = args,
                                          training_params = training_params,
                                          save_path = save_path,
                                          niter_per_ep = 1,
                                          dataset_params = dataset_params)
    if args.is_neptune:
        logger.log_model_summary(pl_finetune)
    
    
    
    # Train the model
    trainer = pl.Trainer(
        default_root_dir = save_path,
        devices = [int(gpu) for gpu in args.gpu.split(',')],
        accelerator = 'gpu',
        strategy = DDPStrategy(find_unused_parameters=True),#'ddp',
        enable_checkpointing = True,
        max_epochs = training_params['num_pretrain_epochs'],
        logger = logger,
        deterministic = True,
        fast_dev_run = False # for development purpose only
        )
        
    test_results = trainer.test(model = pl_finetune,
                                dataloaders = test_loader)
    print(test_results)
    print('Testing completed best model')
    