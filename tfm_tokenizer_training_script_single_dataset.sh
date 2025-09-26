
Dataset='TUEV'
random_seed=5
code_book_size=8192
emb_size=64
experiment_path="./results/tfm_token_${Dataset}" 
# check is the path exists
if [ ! -d $experiment_path ]; then
    mkdir -p $experiment_path
fi


###############################################################################################
# For tokenizer learning, tfm-encoder pretraining and finetuning --> uncomment the following lines
###############################################################################################
vqvae_pretrained_path="${experiment_path}/TFM_TOKENIZER_${Dataset}_2x2x8_${code_book_size}_${emb_size}/tfm_tokenizer_last.pth"
mem_pretrained_path="${experiment_path}/Downstream_Model_MASKED_TOKEN_PREDICTION_PRETRAINING_${Dataset}_${code_book_size}_${emb_size}/tfm_encoder_mtp.pth"
pretrain_gpu='0,1' 
finetune_gpu1='0'
# tfm-tokenizer tokenizer learning
python tfm_tokenizer_token_learning.py --dataset_name $Dataset --code_book_size $code_book_size --emb_size $emb_size --resampling_rate 200 --random_seed $random_seed  --gpu $pretrain_gpu --save_path $experiment_path
# tfm-encoder masked token prediction pretraining
python downstream_transformer_masked_token_prediction_pretraining.py --dataset_name $Dataset --vqvae_pretrained_path $vqvae_pretrained_path --code_book_size $code_book_size --emb_size $emb_size --resampling_rate 200 --random_seed $random_seed --gpu $pretrain_gpu --save_path $experiment_path
# tfm-encoder finetuning
python downstream_transformer_finetuning.py --dataset_name $Dataset --vqvae_pretrained_path $vqvae_pretrained_path --mem_pretrained_path $mem_pretrained_path --code_book_size $code_book_size --emb_size $emb_size  --resampling_rate 200 --random_seed 3 --gpu $finetune_gpu1 --save_path $experiment_path
python downstream_transformer_finetuning.py --dataset_name $Dataset --vqvae_pretrained_path $vqvae_pretrained_path --mem_pretrained_path $mem_pretrained_path --code_book_size $code_book_size --emb_size $emb_size  --resampling_rate 200 --random_seed 3 --gpu $finetune_gpu1 --test --save_path $experiment_path
