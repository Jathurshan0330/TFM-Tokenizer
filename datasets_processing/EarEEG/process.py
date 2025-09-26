import warnings
warnings.filterwarnings('ignore')
import os
import mne
import numpy as np
import pandas as pd
import pickle


# remove the data before onset of sleep scoring
# filter the data to 0.1-100Hz and notch filter 50Hz
# split into 30s segments
# remove data missing corrupted segments

save_path = './earEEG/processed_data/train'
#create the directory if not exists
os.makedirs(save_path, exist_ok=True)


# Training data
sub_id_list= ['001','002','003','004','005','006']
session_list = ['001','002','003','004','005','006','007','008','009','010','011','012']
for sub_id in sub_id_list:
    for session in session_list:
        print(f'Processing sub-{sub_id} ses-{session}')
        #read the data
        try:
            eeg_data = mne.io.read_raw_eeglab(f'./ds005178-download/sub-{sub_id}/ses-{session}/eeg/sub-{sub_id}_ses-{session}_task-sleep_acq-earEEG_eeg.set',preload=True)
        except:
            print(f'Data is corrupted for sub-{sub_id} ses-{session}')
            continue
        #get the data
        sfreq = int(eeg_data.info['sfreq'])
        raw_data = np.array(eeg_data.get_data())
        # print(f'Shape of raw data: {raw_data.shape}')
        # read the scoring event
        try:
            scoring_event = pd.read_csv(f'./ds005178-download/sub-{sub_id}/ses-{session}/eeg/sub-{sub_id}_ses-{session}_task-sleep_acq-scoring_events.tsv', sep='\t')
            print(scoring_event['scoring'].value_counts())
        except:
            print(f'Scoring event is corrupted for sub-{sub_id} ses-{session}')
            continue
        for row_idx, row in scoring_event.iterrows():
            # get the onset of sleep scoring
            onset_of_sleep_scoring = int(row['onset'])*sfreq
            raw_data_segment = raw_data[:,onset_of_sleep_scoring:onset_of_sleep_scoring+30*sfreq]
            
            # check if the data is corrupted
            if np.isnan(raw_data_segment).any():
                print(f'Data is corrupted for sub-{sub_id} ses-{session} row {row_idx}')
                continue
            
            
            
            filtered_data = mne.filter.filter_data(raw_data_segment, sfreq=sfreq, l_freq=0.1, h_freq=100,verbose=False)
            filtered_data = mne.filter.notch_filter(filtered_data, Fs=sfreq, freqs=[50],verbose=False)
           
            
            label = row['scoring_idx']-1
            
            
            
            # save the data as .pkl
            with open(f'{save_path}/sub-{sub_id}_ses-{session}_fft_data_segment_{row_idx}.pkl', 'wb') as f:
                pickle.dump({'X': filtered_data, 'label': label}, f)



# Validation data

save_path = './earEEG/processed_data/val'
#create the directory if not exists
os.makedirs(save_path, exist_ok=True)

sub_id_list= ['007','008']
session_list = ['001','002','003','004','005','006','007','008','009','010','011','012']
for sub_id in sub_id_list:
    for session in session_list:
        print(f'Processing sub-{sub_id} ses-{session}')
        #read the data
        try:
            eeg_data = mne.io.read_raw_eeglab(f'./ds005178-download/sub-{sub_id}/ses-{session}/eeg/sub-{sub_id}_ses-{session}_task-sleep_acq-earEEG_eeg.set',preload=True)
        except:
            print(f'Data is corrupted for sub-{sub_id} ses-{session}')
            continue
        #get the data
        sfreq = int(eeg_data.info['sfreq'])
        raw_data = np.array(eeg_data.get_data())
        # print(f'Shape of raw data: {raw_data.shape}')
        # read the scoring event
        try:
            scoring_event = pd.read_csv(f'./ds005178-download/sub-{sub_id}/ses-{session}/eeg/sub-{sub_id}_ses-{session}_task-sleep_acq-scoring_events.tsv', sep='\t')
            print(scoring_event['scoring'].value_counts())
        except:
            print(f'Scoring event is corrupted for sub-{sub_id} ses-{session}')
            continue
        for row_idx, row in scoring_event.iterrows():
            # get the onset of sleep scoring
            onset_of_sleep_scoring = int(row['onset'])*sfreq
            raw_data_segment = raw_data[:,onset_of_sleep_scoring:onset_of_sleep_scoring+30*sfreq]
            
            # check if the data is corrupted
            if np.isnan(raw_data_segment).any():
                print(f'Data is corrupted for sub-{sub_id} ses-{session} row {row_idx}')
                continue
            
            
            
            filtered_data = mne.filter.filter_data(raw_data_segment, sfreq=sfreq, l_freq=0.1, h_freq=100,verbose=False)
            filtered_data = mne.filter.notch_filter(filtered_data, Fs=sfreq, freqs=[50],verbose=False)
           
            
            label = row['scoring_idx']-1
            
            
            
            # save the data as .pkl
            with open(f'{save_path}/sub-{sub_id}_ses-{session}_fft_data_segment_{row_idx}.pkl', 'wb') as f:
                pickle.dump({'X': filtered_data, 'label': label}, f)


# Testing data
save_path = './earEEG/processed_data/test'
#create the directory if not exists
os.makedirs(save_path, exist_ok=True)

sub_id_list= ['009','010']
session_list = ['001','002','003','004','005','006','007','008','009','010','011','012']
for sub_id in sub_id_list:
    for session in session_list:
        print(f'Processing sub-{sub_id} ses-{session}')
        #read the data
        try:
            eeg_data = mne.io.read_raw_eeglab(f'./ds005178-download/sub-{sub_id}/ses-{session}/eeg/sub-{sub_id}_ses-{session}_task-sleep_acq-earEEG_eeg.set',preload=True)
        except:
            print(f'Data is corrupted for sub-{sub_id} ses-{session}')
            continue
        #get the data
        sfreq = int(eeg_data.info['sfreq'])
        raw_data = np.array(eeg_data.get_data())
        # print(f'Shape of raw data: {raw_data.shape}')
        # read the scoring event
        try:
            scoring_event = pd.read_csv(f'./ds005178-download/sub-{sub_id}/ses-{session}/eeg/sub-{sub_id}_ses-{session}_task-sleep_acq-scoring_events.tsv', sep='\t')
            print(scoring_event['scoring'].value_counts())
        except:
            print(f'Scoring event is corrupted for sub-{sub_id} ses-{session}')
            continue
        for row_idx, row in scoring_event.iterrows():
            # get the onset of sleep scoring
            onset_of_sleep_scoring = int(row['onset'])*sfreq
            raw_data_segment = raw_data[:,onset_of_sleep_scoring:onset_of_sleep_scoring+30*sfreq]
            
            # check if the data is corrupted
            if np.isnan(raw_data_segment).any():
                print(f'Data is corrupted for sub-{sub_id} ses-{session} row {row_idx}')
                continue
            
            
            
            filtered_data = mne.filter.filter_data(raw_data_segment, sfreq=sfreq, l_freq=0.1, h_freq=100,verbose=False)
            filtered_data = mne.filter.notch_filter(filtered_data, Fs=sfreq, freqs=[50],verbose=False)
           
            
            label = row['scoring_idx']-1
            
            
            
            # save the data as .pkl
            with open(f'{save_path}/sub-{sub_id}_ses-{session}_fft_data_segment_{row_idx}.pkl', 'wb') as f:
                pickle.dump({'X': filtered_data, 'label': label}, f)

