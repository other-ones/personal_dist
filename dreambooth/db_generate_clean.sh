export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/cat1";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2736  db_generate_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<cat1>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --resume_unet_path='saved_models/db_models/bigger_seed7777_qlab03_rep1/cat1/db_bigger_qlab03_nomlm_cat1_lr1e6/checkpoints/checkpoint-1000/unet_s1000.pt' \
  --resume_text_encoder_path='saved_models/db_models/bigger_seed7777_qlab03_rep1/cat1/db_bigger_qlab03_nomlm_cat1_lr1e6/checkpoints/checkpoint-1000/text_encoder_s1000.pt' \
  --train_prior_concept1='cat' \
  --eval_prior_concept1='cat' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=dist_results/cat1/baseline \
  --eval_prompt_type='living' \
  --seed=7777 \
  --distill=0 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'


export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/cat1";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 2730  db_generate_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<cat1>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --resume_unet_path='saved_models/db_models/bigger_seed7777_qlab03_rep1/cat1/db_bigger_qlab03_nomlm_cat1_lr1e6/checkpoints/checkpoint-1000/unet_s1000.pt' \
  --resume_text_encoder_path='saved_models/db_models/bigger_seed7777_qlab03_rep1/cat1/db_bigger_qlab03_nomlm_cat1_lr1e6/checkpoints/checkpoint-1000/text_encoder_s1000.pt' \
  --train_prior_concept1='cat' \
  --eval_prior_concept1='cat' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=dist_results/cat1/dist10 \
  --eval_prompt_type='living' \
  --seed=7777 \
  --distill=1 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'


export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/cat1";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 2730  db_generate_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<cat1>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --resume_unet_path='saved_models/db_models/bigger_seed7777_qlab03_rep1/cat1/db_bigger_qlab03_nomlm_cat1_lr1e6/checkpoints/checkpoint-1000/unet_s1000.pt' \
  --resume_text_encoder_path='saved_models/db_models/bigger_seed7777_qlab03_rep1/cat1/db_bigger_qlab03_nomlm_cat1_lr1e6/checkpoints/checkpoint-1000/text_encoder_s1000.pt' \
  --train_prior_concept1='cat' \
  --eval_prior_concept1='cat' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=dist_results/cat1/dist05 \
  --eval_prompt_type='living' \
  --seed=7777 \
  --distill=0.5 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'




export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/cat1";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 2732  db_generate_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<cat1>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --resume_unet_path='saved_models/db_models/bigger_seed7777_qlab03_rep1/cat1/db_bigger_qlab03_nomlm_cat1_lr1e6/checkpoints/checkpoint-1000/unet_s1000.pt' \
  --resume_text_encoder_path='saved_models/db_models/bigger_seed7777_qlab03_rep1/cat1/db_bigger_qlab03_nomlm_cat1_lr1e6/checkpoints/checkpoint-1000/text_encoder_s1000.pt' \
  --train_prior_concept1='cat' \
  --eval_prior_concept1='cat' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=tmp_dist05 \
  --eval_prompt_type='living' \
  --seed=7777 \
  --distill=0.5 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'

