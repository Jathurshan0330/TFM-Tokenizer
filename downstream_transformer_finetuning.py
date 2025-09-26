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

class Pl_downstream_transformer_finetuning(pl.LightningModule):
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
        self.vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path,map_location=self.device))
        self.vqvae.to(self.device)
        self.vqvae.eval()

        print(f'Loading TFM-TOKEN classifier model: 64x4')
        self.tfm_token = get_tfm_token_classifier_64x4(n_classes=self.dataset_params['num_classes'],code_book_size=args.code_book_size, emb_size=args.emb_size)
        
        print('Loading the pretrained checkpoint...')
        print(f'Loading the pretrained model from: {args.mem_pretrained_path}')
        checkpoint = torch.load(args.mem_pretrained_path, map_location=self.device)#map_location="cpu")
        filtered_checkpoint = {
            key: value
            for key, value in checkpoint.items()
            if "classification_head" not in key
        }
        self.tfm_token.load_state_dict(filtered_checkpoint, strict=False)
        self.tfm_token.to(self.device)
        
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
        
    def configure_optimizers(self):
        # ADAMW optimizer
        if self.training_params['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.tfm_token.parameters(), 
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
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        pred = self(x)
        
        if self.dataset_params['classification_task']=='binary':
            if self.args.dataset_name == 'CHBMIT':
                loss = focal_loss(pred, y)
            else:
                loss = BCE(pred, y)
        elif self.dataset_params['classification_task']=='multi_class':
            loss_fn = LabelSmoothingCrossEntropy(smoothing = self.training_params['label_smoothing'])
            loss = loss_fn(pred,y)
            

        self.log('train_step_loss',loss,prog_bar=True,sync_dist=True)
        
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        with torch.no_grad():
            pred = self(x)
            if self.dataset_params['classification_task']=='binary':
                if self.args.dataset_name == 'CHBMIT':
                    loss = focal_loss(pred, y)
                else:
                    loss = BCE(pred, y)
                pred = torch.sigmoid(pred)
            elif self.dataset_params['classification_task']=='multi_class':
                loss_fn = LabelSmoothingCrossEntropy(smoothing = self.training_params['label_smoothing'])
                
                loss = loss_fn(pred,y)
            self.log('val_step_loss',loss,prog_bar=True,sync_dist=True)
            
            step_pred = pred.cpu().numpy()
            step_y = y.cpu().numpy()
            self.val_step_outputs.append([step_pred,step_y])
        return loss
    
    def on_validation_epoch_end(self):
        val_pred = np.concatenate([i[0] for i in self.val_step_outputs])
        val_y = np.concatenate([i[1] for i in self.val_step_outputs])
        
        if self.dataset_params['classification_task']=='binary':
            is_binary = True
        else:
            is_binary = False
            
        results = get_metrics(np.array(val_pred), np.array(val_y), self.metrics, is_binary)
        
        for i in range(len(self.metrics)):
            if i == 2:
                self.log(f'val_{self.metrics[i]}',results[self.metrics[i]],prog_bar=True,sync_dist=True)
            else:
                self.log(f'val_{self.metrics[i]}',results[self.metrics[i]],prog_bar=False,sync_dist=True)
                
        self.val_step_outputs = []
        
        return results
    
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
    parser.add_argument('--mem_pretrained_path', type=str, default=None, help='Path to the pretrained model')
    parser.add_argument('--code_book_size', type=int, default=8192, help='Code book size')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--is_neptune', type=bool, default=False, help='Log to neptune')
    
    parser.add_argument('--save_path', type=str, default=None, help='Save path')
    parser.add_argument('--random_seed', type=int, default=5, help='Random seed')
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
    
    experiment_name = f'TFM_ENCODER_FINETUNING_{args.dataset_name}_random_seed_{args.random_seed}'
    
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
        
    val_loader = get_dataloaders(data_name= args.dataset_name,
                                   train_val_test='val',
                                   resampling_rate=args.resampling_rate,
                                   batch_size=training_params['batch_size'], 
                                   num_workers=8,
                                   signal_transform = None,
                                   random_seed=args.random_seed)
    
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
    
    
    
    pl_finetune = Pl_downstream_transformer_finetuning(args = args,
                                          training_params = training_params,
                                          save_path = save_path,
                                          niter_per_ep = niter_per_ep,
                                          dataset_params = dataset_params)
    if args.is_neptune:
        logger.log_model_summary(pl_finetune)
    
    # Callbacks
    if dataset_params['classification_task']=='binary':
        
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, 
                                            save_top_k=1, 
                                            monitor="val_roc_auc",
                                            mode="max", 
                                            filename="best_model",
                                            save_weights_only=True,
                                            save_last = True)
    
    else:
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, 
                                            save_top_k=1, 
                                            monitor="val_cohen_kappa",
                                            mode="max", 
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
        fast_dev_run = False # for development purpose only
        )
        
    if args.test_only:
        saved_checkpoint_path = os.path.join(save_path,'best_model.ckpt')
    
        test_results = trainer.test(model = pl_finetune,
                                    ckpt_path = saved_checkpoint_path,
                                    dataloaders = test_loader)
        print(test_results)
        print('Testing completed best model')
    else:
        trainer.fit(pl_finetune, 
                    train_dataloaders = train_loader,
                    val_dataloaders = val_loader)
        print('Training completed')
        
        # testing the classifier
        test_results = trainer.test(model = pl_finetune,
                                    ckpt_path = 'best',
                                    dataloaders = test_loader)
        
        print('Testing completed')
        
        # save the best model
        best_model_path = os.path.join(save_path,'best_model.ckpt')
        pl_finetune = Pl_downstream_transformer_finetuning.load_from_checkpoint(best_model_path,
                                                                    args = args,
                                                                    training_params = training_params,
                                                                    save_path = save_path,
                                                                    niter_per_ep = niter_per_ep,
                                                                    dataset_params = dataset_params)
        torch.save(pl_finetune.tfm_token.state_dict(),os.path.join(save_path,'best_model.pth'))
        print('Model saved')