export CUDA_VISIBLE_DEVICES=7;
python visualize_attentions.py \
--placeholder_token1='<cat1>' \
--train_prior_concept1='cat' \
--prompt='a picture of {} with blue house in the background' \
--dst_dir='attn_maps/custom_special' \
--resume_text_encoder_path='saved_models/db_models/bigger_seed7777_qlab03_rep1/cat1/db_bigger_qlab03_mlm00001_cat1_mprob015_mbatch25_mtarget_masked_lr1e6/checkpoints/checkpoint-1000/text_encoder_s1000.pt' \
--mask_tokens='[MASK]'



python visualize_attentions.py \
--placeholder_token1='<cat1>' \
--train_prior_concept1='cat' \
--prompt='a picture of cute cat with blue house in the background' \
--dst_dir='attn_maps/prior_special' \
--mask_tokens='[MASK]'




# python visualize_attentions.py \
# --placeholder_token1='<cat1>' \
# --train_prior_concept1='dog' \
# --prompt='a picture of {} in the snow' \
# --dst_dir='attn_maps/custom_mlm00001' \
# --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/cat1/ti_bigger_qlab03_prior_mlm00001_cat1_mprob015_mbatch25_mtarget_masked/checkpoints/learned_embeds_s3000.pt'
