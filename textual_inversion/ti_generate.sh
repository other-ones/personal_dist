export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2736  ti_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=tmp_prior \
  --eval_prompt_type='living' \
  --seed=7777 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'



export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2736  ti_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=tmp_baseline \
  --eval_prompt_type='living' \
  --seed=7777 \
  --distill=1 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'

export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 2732  ti_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=tmp_dist05 \
  --eval_prompt_type='living' \
  --seed=7777 \
  --distill=0.5 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'


export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 2733  ti_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=tmp_dist02 \
  --eval_prompt_type='living' \
  --seed=7777 \
  --distill=0.2 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'

export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=4;
accelerate launch --main_process_port 2734  ti_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=tmp_dist0 \
  --eval_prompt_type='living' \
  --seed=7777 \
  --distill=0 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'

export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 2732  ti_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_cat1>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/pet_cat1/ti_bigger_qlab03_prior_nomlm_pet_cat1/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='cat' \
  --eval_prior_concept1='cat' \
  --include_prior_concept=1 \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=tmp_pet_cat1 \
  --eval_prompt_type='living' \
  --seed=7777 \
  --distill=0.5 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'
