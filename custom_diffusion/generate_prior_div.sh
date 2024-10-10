export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=3;
accelerate launch generate_prior_div.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --resolution=512  \
  --eval_batch_size=15 \
  --train_prior_concept1='a dog' \
  --eval_prior_concept1='a dog' \
  --eval_prompt_type='living' \
  --caption_path='../datasets_pkgs/captions/prior/captions_prior_pet.txt' \
  --dst_exp_path="results/prior_div/dog"

