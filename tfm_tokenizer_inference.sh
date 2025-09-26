# Inference to test our TFM-Tokenizer using the pre-trained weigths

Dataset='TUEV'
random_seed=5
code_book_size=8192
emb_size=64
# check is the path exists
if [ ! -d $experiment_path ]; then
    mkdir -p $experiment_path
fi

#######################################################
# For Inference and Evaluation using pre-trained models trained under multi-dataset setting --> uncomment the following lines
#######################################################
experiment_path="./results/tfm_token_testing_${Dataset}_multi"
vqvae_pretrained_path="./pretrained_weigths/multiple_dataset_settings/Pretrained_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth"
finetuned_path="./pretrained_weigths/multiple_dataset_settings/Finetuned_${Dataset}/tfm_encoder_best_model.pth"
finetune_gpu1='0'
# Inference
python tfm_tokenizer_inference.py --dataset_name $Dataset --vqvae_pretrained_path $vqvae_pretrained_path --finetuned_path $finetuned_path --code_book_size $code_book_size --emb_size $emb_size --resampling_rate 200 --random_seed $random_seed --gpu $finetune_gpu1 --save_path $experiment_path


########################################################
# For Inference and Evaluation using models trained under single dataset setting --> uncomment the following lines
########################################################
# experiment_path="./results/tfm_token_testing_${Dataset}_single"
# vqvae_pretrained_path="./pretrained_weigths/single_dataset_settings/${Dataset}_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth"
# finetuned_path="./pretrained_weigths/single_dataset_settings/${Dataset}_tfm_tokenizer_2x2x8/tfm_encoder_best_model.pth"
# finetune_gpu1='0'
# # Inference
# python tfm_tokenizer_inference.py --dataset_name $Dataset --vqvae_pretrained_path $vqvae_pretrained_path --finetuned_path $finetuned_path --code_book_size $code_book_size --emb_size $emb_size --resampling_rate 200 --random_seed $random_seed --gpu $finetune_gpu1 --save_path $experiment_path

