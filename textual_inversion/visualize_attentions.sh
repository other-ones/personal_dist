# random
export CUDA_VISIBLE_DEVICES=0;
python visualize_attentions.py \
--placeholder_token1='<dog6>' \
--train_prior_concept1='dog' \
--prompt='a picture of {} with blue house in the background' \
--dst_dir='attn_maps/custom_special_rand2' \
--learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
--distill=-2 \
--include_special
export CUDA_VISIBLE_DEVICES=0;
python visualize_attentions.py \
--placeholder_token1='<dog6>' \
--train_prior_concept1='dog' \
--prompt='a picture of {} with blue house in the background' \
--dst_dir='attn_maps/custom_all_rand2' \
--learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
--distill=-2 \
--include_all


# baseline
export CUDA_VISIBLE_DEVICES=0;
python visualize_attentions.py \
--placeholder_token1='<dog6>' \
--train_prior_concept1='dog' \
--prompt='a picture of {} with blue house in the background' \
--dst_dir='attn_maps/custom_baseline_special_dist01' \
--learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
--distill=0.1 \
--include_special

export CUDA_VISIBLE_DEVICES=0;
python visualize_attentions.py \
--placeholder_token1='<dog6>' \
--train_prior_concept1='dog' \
--prompt='a picture of {} with blue house in the background' \
--dst_dir='attn_maps/custom_baseline_all' \
--learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
--distill=0 \
--include_all
