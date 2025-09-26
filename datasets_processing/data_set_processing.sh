
CHBMIT_PATH='[PATH_TO_CHBMIT_DATASET]'

TUAB_PATH='[PATH_TO_TUAB_DATASET]'
TUAB_PATH_SAVE='[PATH_TO_SAVE_TUAB_DATASET]'

TUEV_PATH='[PATH_TO_TUEV_DATASET]'
TUEV_PATH_SAVE='[PATH_TO_SAVE_TUEV_DATASET]'

cd ./datasets_processing






# process TUAB
python './TUAB/process.py' --root $TUAB_PATH --root_save $TUAB_PATH_SAVE

# process TUEV
python './TUEV/process.py' --root $TUEV_PATH --root_save $TUEV_PATH_SAVE


# Process CHB-MIT dataset
python './CHB-MIT/process_1.py' --signals_path $CHBMIT_PATH 
python './CHB-MIT/process_2.py' --signals_path $CHBMIT_PATH
# clean segments will be saved at $CHBMIT_PATH/clean_segments

# Process EarEEG dataset
python './EarEEG/process.py' 