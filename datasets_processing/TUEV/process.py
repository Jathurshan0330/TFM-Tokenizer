import mne
import numpy as np
import random
import os
import pickle
from tqdm import tqdm
import argparse

## fixing seed
seed = 5
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 256.0
    [numChan, numPoints] = signals.shape
    # for i in range(numChan):  # standardize each channel
    #     if np.std(signals[i, :]) > 0:
    #         signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]]
            - signals[signal_names["EEG F7-REF"]],  # 0
            (
                signals[signal_names["EEG F7-REF"]]
                - signals[signal_names["EEG T3-REF"]]
            ),  # 1
            (
                signals[signal_names["EEG T3-REF"]]
                - signals[signal_names["EEG T5-REF"]]
            ),  # 2
            (
                signals[signal_names["EEG T5-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 3
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F8-REF"]]
            ),  # 4
            (
                signals[signal_names["EEG F8-REF"]]
                - signals[signal_names["EEG T4-REF"]]
            ),  # 5
            (
                signals[signal_names["EEG T4-REF"]]
                - signals[signal_names["EEG T6-REF"]]
            ),  # 6
            (
                signals[signal_names["EEG T6-REF"]]
                - signals[signal_names["EEG O2-REF"]]
            ),  # 7
            (
                signals[signal_names["EEG FP1-REF"]]
                - signals[signal_names["EEG F3-REF"]]
            ),  # 14
            (
                signals[signal_names["EEG F3-REF"]]
                - signals[signal_names["EEG C3-REF"]]
            ),  # 15
            (
                signals[signal_names["EEG C3-REF"]]
                - signals[signal_names["EEG P3-REF"]]
            ),  # 16
            (
                signals[signal_names["EEG P3-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 17
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F4-REF"]]
            ),  # 18
            (
                signals[signal_names["EEG F4-REF"]]
                - signals[signal_names["EEG C4-REF"]]
            ),  # 19
            (
                signals[signal_names["EEG C4-REF"]]
                - signals[signal_names["EEG P4-REF"]]
            ),  # 20
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
        )
    )  # 21
    return new_signals


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName,preload = True)
    # ######## apply filtering  0.1-75 Hz ################
    # Rawdata = Rawdata.copy().filter(l_freq = 0.1, h_freq = 75)
    # ######## apply  notch filtering  50 Hz ################
    # Rawdata = Rawdata.copy().notch_filter(freqs = 50)
    
    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(256, n_jobs=5)
    
    # signals, times = Rawdata[:]
    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    
    
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  # event is the .rec file in the form of an array
                    signals = convert_signals(signals, Rawdata)
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue
                signals, offending_channels, labels = BuildEvents(signals, times, event)

                for idx, (signal, offending_channel, label) in enumerate(
                    zip(signals, offending_channels, labels)
                ):
                    sample = {
                        "signal": signal,
                        "offending_channel": offending_channel,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


"""
TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
"""
# root to TUEV dataset
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='', help='Path to TUAB dataset')
parser.add_argument('--root_save', type=str, default='', help='Path to save processed data')

args = parser.parse_args()
root = args.root
root_save = args.root_save


train_out_dir = os.path.join(root_save, "processed_train")
eval_out_dir = os.path.join(root_save, "processed_eval")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)

BaseDirTrain = os.path.join(root, "train")
fs = 256
TrainFeatures = np.empty(
    (0, 16, fs)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TrainLabels = np.empty([0, 1])
TrainOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir
)

BaseDirEval = os.path.join(root, "eval")
fs = 256
EvalFeatures = np.empty(
    (0, 16, fs)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
EvalLabels = np.empty([0, 1])
EvalOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir
)


# split train and val 

train_files = os.listdir(train_out_dir)
train_sub = list(set([f.split("_")[0] for f in train_files]))
print('Number of subjects:', len(train_sub))
val_sub = np.random.choice(train_sub, size=int(len(train_sub)*0.2), replace=False)
train_sub = list(set(train_sub) - set(val_sub))
print('Number of subjects in train:', len(train_sub), 'Number of subjects in val:', len(val_sub))
val_files = [f for f in train_files if f.split("_")[0] in val_sub]
train_files = [f for f in train_files if f.split("_")[0] in train_sub]
print('Number of recordings in train:', len(train_files), 'Number of recordings in val:', len(val_files))


train_split_path = os.path.join(train_out_dir, "processed_train_split")
val_split_path = os.path.join(train_out_dir, "processed_val_split")

if not os.path.exists(train_split_path):
    os.makedirs(train_split_path)
if not os.path.exists(val_split_path):
    os.makedirs(val_split_path)
    
for file in train_files:
    os.system(f"cp {os.path.join(train_out_dir, file)} {os.path.join(train_split_path, file)}")
for file in val_files:
    os.system(f"cp {os.path.join(train_out_dir, file)} {os.path.join(val_split_path, file)}")

num_sub_in_eval = list(set([f.split("_")[0] for f in os.listdir(eval_out_dir)]))
#create a text file in root_save to store the count
with open(os.path.join(root_save, 'count.txt'), 'w') as f:
    f.write(f'Number of subjects in train: {len(train_sub)}\nNumber of subjects in val: {len(val_sub)}\nNumber of recordings in train: {len(train_files)}\nNumber of recordings in val: {len(val_files)}')
    f.write(f'Number of subjects in eval: {len(num_sub_in_eval)}, Number of recordings in eval: {len(os.listdir(eval_out_dir))}')
    f.close()