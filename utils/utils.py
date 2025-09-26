import os
import random
import numpy as np
import torch
import math
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
from scipy import signal
import matplotlib.pyplot as plt 

from einops import rearrange

def seed_everything(seed=5):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    




def read_hyp_params(hyp_path):
    with open(hyp_path, 'r') as f:
        hyp = f.readlines()
    hyp = [i.strip() for i in hyp]
    hyp = [i.split(':') for i in hyp]
    hyp = {i[0].strip():i[1].strip() for i in hyp}
    
    for key in hyp.keys():
        if hyp[key].lower() == 'true':
            hyp[key] = True
        elif hyp[key].lower() == 'false':
            hyp[key] = False
        else:
            try:
                hyp[key] = int(hyp[key])
            except:
                try:
                    hyp[key] = float(hyp[key])
                except:
                    try:
                        hyp[key] = eval(hyp[key])
                    except:
                        continue 
    # convert the dict to a object
    class Hyp():
        def __init__(self, **entries):
            self.__dict__.update(entries)
    hyp = Hyp(**hyp)                    
    return hyp
        
        

def BCE(y_hat, y):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()
        
        
# define focal loss on binary classification for CHB-MIT
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    # y_hat: (N, 1)
    # y: (N, 1)
    # alpha: float
    # gamma: float
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()

def weighted_sum_mse_loss(predictions, targets, weights):
    if weights.shape != predictions.shape:
        weights = weights.view_as(predictions)
    
    # Compute element-wise squared errors
    squared_errors = (predictions - targets) ** 2
    
    # Apply weights to the squared errors
    weighted_squared_errors = squared_errors * weights
    
    # Compute the mean of the weighted squared errors
    loss = torch.mean(weighted_squared_errors)

    return loss
    
    
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results


def get_stft(X,resampling_rate = 200):
        window_len = resampling_rate # 1s
        hop_len = resampling_rate//2 #0.5s
        X_fft = []
        for i in range(X.shape[0]):
            f, _, Zxx = signal.stft(X[i], 
                        fs=resampling_rate, 
                        nperseg=window_len, 
                        noverlap=window_len-hop_len,
                        return_onesided=True,boundary=None,
                        padded = False, scaling='spectrum')
            X_fft.append(np.abs(Zxx))
        X = np.stack(X_fft, axis=0) # (batch, channels, freqs, timesamples)
        X = X[:,:,:-1,:]
        X = torch.FloatTensor(X)
        return X 
    
def get_stft_torch(X, resampling_rate = 200):
    B,C,T = X.shape
    x_temp = rearrange(X, 'B C T -> (B C) T')
    window = torch.hann_window(resampling_rate).to(x_temp.device)
    x_stft_temp = torch.abs(torch.stft(x_temp, n_fft=resampling_rate, hop_length=resampling_rate//2, 
                          onesided = True,
                          return_complex=True, center = False,#normalized = True,
                          window = window)[:,:resampling_rate//2,:])
    
    x_stft_temp = rearrange(x_stft_temp, '(B C) F T -> B C F T', B=B)
    
    return x_stft_temp


def plot_token_interpret_tuev(x_temp,x_stft,x_tokens,label,intersection):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6),dpi=200)
    classes = ['SPSW','GPED','PLED','EYEM','ARTF','BCKG']
    class_full_name = {'SPSW':"Spike and Sharp Wave",
                     'GPED':"Generalized Periodic Epileptiform Discharges",
                     'PLED':"Periodic Lateralized Epileptiform Discharges",
                     'EYEM':"Eye Movement",
                     'ARTF':"Artifact",
                     'BCKG':"Background"}
    label_class = classes[label]
    label_class_full = class_full_name[label_class]
    
    fig.suptitle(f'{label_class_full} - tokens to watch {intersection}', fontsize=12, fontweight = 'bold')

    ax[0].plot(x_temp[0, :], color = 'black', linewidth = 2)
    ax[0].set_xlim(0, x_temp.shape[1])
    # if intersection in x_tokens[i]:
        # ax[0].set_title(x_tokens[i])
    tick_positions = np.arange(100, len(x_temp[0,:]), 100)

    # Create a list of tokens to display at each tick position
    time_labels = np.arange(0.5, 5.5, 0.5)

    # Ensure time_labels matches tick_positions length
    if len(time_labels) > len(tick_positions):
        time_labels = time_labels[:len(tick_positions)]
        
    tokens_display = []
    for i, pos in enumerate(tick_positions):
        if i < len(x_tokens):
            tokens_display.append(f"{time_labels[i]:.1f}s\n\n{x_tokens[i]}")
        else:
            tokens_display.append('')

    ax[0].set_xticks(tick_positions)
    x_tokens_temp = []

    ax[0].minorticks_on()
    ax[0].set_xticklabels(tokens_display, fontsize = 14)
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])
    ax[0].tick_params(axis='both', which='major', width=2, length=10)
    ax[0].tick_params(axis='both', which='minor', width=2, length=5)
    ax[0].set_xlabel('', fontsize=0)
    ax[0].set_ylabel('', fontsize=0)
    ax[0].set_title('', fontsize=0)

    ax[0].spines['bottom'].set_color('black')
    ax[0].spines['bottom'].set_linewidth(3)
    ax[0].spines['left'].set_visible(False)#.set_color('black')
    # ax[0].spines['left'].set_linewidth(3)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    # Color-coded regions:
    # Each region is length=200 with an overlap of 100
    region_length = 200
    overlap       = 100
    signal_length = len(x_temp[0,:])


    ax[1].imshow(x_stft[0,:40,:], aspect='auto',origin='lower', cmap='jet',interpolation='bilinear')
    # if intersection in x_tokens[i]:
    ax[1].set_xlabel('', fontsize=0)
    ax[1].set_ylabel('', fontsize=0)
    ax[1].set_title('', fontsize=0)



    tick_positions = np.arange(100, len(x_temp[0,:])+1, 100)
    # Create a list of tokens to display at each tick position
    time_labels = np.arange(0., 5.5, 0.5)

    # Ensure time_labels matches tick_positions length
    if len(time_labels) > len(tick_positions):
        time_labels = time_labels[:len(tick_positions)]
    # print(time_labels)
    tokens_display = []
    for i, pos in enumerate(tick_positions):
        # if i==0:
        #     continue
        if i < len(x_tokens)+1:
            tokens_display.append(f"{time_labels[i]:.1f}s\n\n{x_tokens[i-1]}")
        else:
            tokens_display.append('')



    ax[1].minorticks_on()
    ax[1].set_xticklabels(tokens_display, fontsize = 14)
    ax[1].set_yticks([])
    ax[1].set_yticklabels([])
    ax[1].tick_params(axis='both', which='major', width=2, length=10)
    ax[1].tick_params(axis='both', which='minor', width=2, length=5)
    ax[1].set_xlabel('', fontsize=0)
    ax[1].set_ylabel('', fontsize=0)
    ax[1].set_title('', fontsize=0)
    ax[1].spines['bottom'].set_color('black')
    ax[1].spines['bottom'].set_linewidth(3)
    ax[1].spines['left'].set_visible(False)#.set_color('black')
    # ax[0].spines['left'].set_linewidth(3)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)


    plt.tight_layout()
    plt.show()