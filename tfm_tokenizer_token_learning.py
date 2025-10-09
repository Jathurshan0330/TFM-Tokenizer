import warnings

warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
import torch
import yaml

import lightning as pl
from lightning.pytorch.loggers import NeptuneLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from utils.utils import seed_everything, cosine_scheduler, weighted_sum_mse_loss
from utils.utils import get_stft_torch 
from datasets.data_loaders import get_dataloaders, get_pretrain_dataloaders

from einops import rearrange
from models.tfm_token import  freq_bin_temporal_masking
from models.tfm_token import get_tfm_tokenizer_2x2x8




class Pl_tfm_tokenizer_token_learning(pl.LightningModule):
    def __init__(self,args,training_params,save_path,niter_per_ep):
        super().__init__()
        self.args = args
        self.training_params = training_params
        self.save_path = save_path
        self.niter_per_ep = niter_per_ep
        
        print('Loading tfm-token tokenizer model...')
        
        print('Loading tfm-tokenizer 2x2x8 model...')
        self.vqvae = get_tfm_tokenizer_2x2x8(code_book_size=args.code_book_size, emb_size=args.emb_size)
        
        trainable_parameters = sum(p.numel() for p in self.vqvae.parameters() if p.requires_grad)
        fixed_parameters = sum(p.numel() for p in self.vqvae.parameters() if not p.requires_grad)
        with open(os.path.join(self.save_path,'tfm_tokenizer_model.txt'),'w') as f:
            f.write('Number of trainable parameters: \n')
            f.write(str(trainable_parameters))
            f.write('\n')
            f.write('Number of fixed parameters: \n')
            f.write(str(fixed_parameters))
            f.write('\n')
            f.write(str(self.vqvae))
            f.close()
    
    def configure_optimizers(self):
        # ADAMW optimizer
        if self.training_params['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.vqvae.parameters(), 
                lr=self.training_params['lr'], 
                weight_decay=self.training_params['weight_decay'],
                betas = (self.training_params['beta1'],self.training_params['beta2'])
            )
        
        # Compute the scheduler
        epochs = self.training_params['num_pretrain_epochs']
        warmup_epochs = self.training_params.get('warmup_epochs', 0)
        scheduler_values = cosine_scheduler(
            base_value=self.training_params['lr'],
            final_value=self.training_params.get('final_lr', 0),
            epochs=epochs,
            niter_per_ep=self.niter_per_ep,
            warmup_epochs=warmup_epochs,
        )
        
        # save the scheduler values
        with open(os.path.join(self.save_path,'scheduler_values.txt'),'w') as f:
            f.write('Scheduler values: \n')
            for i in range(len(scheduler_values)):
                f.write(str(scheduler_values[i]))
                f.write('\n')
            f.close()
        
        # Wrap the scheduler in LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: scheduler_values[step]/self.training_params['lr'] if step < len(scheduler_values) else scheduler_values[-1]/self.training_params['lr'],
        )      
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    

    def forward(self,x):
        x_temporal = x 
        # apply STFT
        x = get_stft_torch(x_temporal, resampling_rate = self.args.resampling_rate)
        x = rearrange(x,'B C F T -> (B C) F T').to(x_temporal.device)
        x_temporal = rearrange(x_temporal,'B C T -> (B C) T')
        
        # apply frequency bin masking and temporal masking
        x_masked, x_masked_sym, mask, masked_sym = freq_bin_temporal_masking(x, freq_mask_ratio=0.5, 
                                                            freq_bin_size=5, 
                                                            time_mask_ratio=0.5,#.5, 
                                                            time_bin_size=1)
        
        recon_out, _, quant_out, quant_in = self.vqvae(x_masked,x_temporal)
        recon_out_sym, _, quant_out_sym, quant_in_sym = self.vqvae(x_masked_sym,x_temporal)
        
        quant_loss, _, _ = self.vqvae.vec_quantizer_loss(quant_in,quant_out)
        quant_sym_loss, _, _ = self.vqvae.vec_quantizer_loss(quant_in_sym,quant_out_sym)
        
        weigths = torch.ones_like(x).detach()
        recon_loss =  weighted_sum_mse_loss(recon_out,x,weigths)
        recon_loss_sym =  weighted_sum_mse_loss(recon_out_sym,x,weigths)
        
        recon_loss = recon_loss + recon_loss_sym
        quant_loss = quant_loss + quant_sym_loss
        
        return recon_loss, quant_loss
    
    def train_step_pretrain(self,train_batch,batch_idx):
        tuab, tuev, chbmit, iiic = train_batch
        x_16_10_list = []
        recon_loss = 0
        quant_loss = 0
        
        if len(tuab) > 0:
            x_16_10_list.append(tuab)
        if len(chbmit) > 0:
            x_16_10_list.append(chbmit)
        if len(iiic) > 0:
            x_16_10_list.append(iiic)
        if len(x_16_10_list) > 0:
            x_16_10 = torch.cat(x_16_10_list,dim=0)
            recon_loss_temp, quant_loss_temp = self(x_16_10)
            recon_loss += recon_loss_temp
            quant_loss += quant_loss_temp
        
        if len(tuev) > 0:
            recon_loss_temp, quant_loss_temp = self(tuev)
            recon_loss += recon_loss_temp
            quant_loss += quant_loss_temp

            
        return recon_loss, quant_loss
        
    def training_step(self, train_batch, batch_idx):
        if self.args.dataset_name == 'pretrain':
            recon_loss, quant_loss = self.train_step_pretrain(train_batch,batch_idx)
            loss = recon_loss + quant_loss
            self.log('train_step_loss', loss, prog_bar=True, sync_dist=True)
            self.log('train_step_quant_loss', quant_loss, prog_bar=False, sync_dist=True)
            self.log('train_step_rec_loss', recon_loss, prog_bar=False, sync_dist=True)

        else:
            x,y = train_batch
            recon_loss, quant_loss = self(x)
            loss = recon_loss + quant_loss
            self.log('train_step_loss', loss, prog_bar=True, sync_dist=True)
            self.log('train_step_quant_loss', quant_loss, prog_bar=False, sync_dist=True)
            self.log('train_step_rec_loss', recon_loss, prog_bar=False, sync_dist=True)
        
        return loss
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='pretrain', help='Dataset name')
    parser.add_argument('--pretraining_dataset_list', type=str, default='TUAB,TUEV,CHBMIT,IIIC', help='Pretraining dataset list')
    parser.add_argument('--resampling_rate', type=int, default=200, help='Resampling rate')
    parser.add_argument('--gpu',type=str, default=None, help='GPU to use')
    
    parser.add_argument('--code_book_size', type=int, default=8192, help='Code book size')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding size')  
    parser.add_argument('--random_seed', type=int, default=5, help='Random seed')
    parser.add_argument('--is_neptune', type=bool, default=False, help='Log to neptune')
    
    parser.add_argument('--save_path', type=str, default=None, help='Save path')
    args = parser.parse_args()
    
    # read the configuration file
    with open("./configs/tfm_tokenizer_training_configs.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
        
    # training parameters
    training_params = config[f'tokenizer_training_{args.dataset_name}']

    experiment_name = f'TFM_TOKENIZER_{args.dataset_name}_2x2x8_{args.code_book_size}_{args.emb_size}'
    
    
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
        f.close()
        
    # Create the dataloaders
    seed_everything(args.random_seed)
    
    if args.dataset_name == 'pretrain':
        dataset_list = args.pretraining_dataset_list.split(',')
        train_loader =  get_pretrain_dataloaders(
                                   resampling_rate=args.resampling_rate,
                                   batch_size=training_params['batch_size'],
                                   num_workers=8,
                                   signal_transform = None,
                                   dataset_list = dataset_list,
                                   random_seed=args.random_seed)
        print('Testing the dataloaders')
        for i, (tuab, tuev, chbmit, iiic) in enumerate(train_loader):
            print(f'Batch {i}: {tuab.shape}, {tuev.shape}, {chbmit.shape}, {iiic.shape}')
            break
    else:
        train_loader = get_dataloaders(data_name= args.dataset_name,
                                    train_val_test='train',
                                    resampling_rate=args.resampling_rate,
                                    batch_size=training_params['batch_size'], 
                                    num_workers=8,
                                    signal_transform = None,
                                    random_seed=args.random_seed)
    
        print('Testing the dataloaders')
        for i, (X, _) in enumerate(train_loader):
            print(f'Batch {i}: {X.shape}')
            break
    niter_per_ep = len(train_loader)
    if args.gpu:  
        devices = [int(gpu) for gpu in args.gpu.split(',')]
        niter_per_ep = niter_per_ep//len(devices)
        
    
    # initializing logger
    if args.is_neptune:
        logger = NeptuneLogger(
            api_key="[APIKEY]",  # replace with your own
            project="[PROJECT_NAME]",  # replace with your own
            tags=[experiment_name],
            log_model_checkpoints=True,  # save checkpoints
        )
    else: 
        logger = CSVLogger(save_dir = save_path,name = 'logs')
    
    # tokenizer pretraining
    pl_vqvae = Pl_tfm_tokenizer_token_learning(args = args,training_params = training_params,save_path = save_path,niter_per_ep = niter_per_ep)
    if args.is_neptune:
        logger.log_model_summary(pl_vqvae)
    

    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, 
                                          save_top_k=1, 
                                          monitor="train_step_loss",
                                          mode="min", 
                                          filename="best_model",
                                          save_weights_only=True,
                                          save_last = True)
    
    
    
    # Train the model

    trainer = pl.Trainer(
        default_root_dir = save_path,
        devices = [int(gpu) for gpu in args.gpu.split(',')],
        accelerator = 'gpu',
        strategy = DDPStrategy(find_unused_parameters=True),#'ddp',
        enable_checkpointing = True,
        callbacks = [checkpoint_callback],
        max_epochs = training_params['num_pretrain_epochs'],
        logger = logger,
        deterministic = True,
        fast_dev_run = False ,# for development purpose only
        precision='bf16'
    )
    
    trainer.fit(pl_vqvae, 
                train_dataloaders=train_loader)
    
    print('Training completed!')

    # load and save last model 
    ##############
    # We utilized the last saved model checkpoint as the tokenizer model
    ##############
    last_model_path = os.path.join(save_path,'last.ckpt')
    pl_vqvae = Pl_tfm_tokenizer_token_learning.load_from_checkpoint(last_model_path,
                                                        args = args,
                                                        training_params = training_params,
                                                        save_path = save_path, niter_per_ep = niter_per_ep)
                                                       
    torch.save(pl_vqvae.vqvae.state_dict(),os.path.join(save_path,'tfm_tokenizer_last.pth'))
    
    print('Model saved!')
    
    
    
            
