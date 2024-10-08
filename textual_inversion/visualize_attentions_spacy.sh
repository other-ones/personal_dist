python visualize_attentions_spacy.py \
--placeholder_token1='<dog6>' \
--train_prior_concept1='dog' \
--prompt='a picture of {} with blue house in the background' \
--dst_dir='attn_maps/custom_special' \
--learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
--include_special



python visualize_attentions_spacy.py \
--placeholder_token1='<dog6>' \
--train_prior_concept1='dog' \
--prompt='a picture of a cute dog that is red' \
--dst_dir='attn_maps/prior_special' \
--include_special



# python visualize_attentions.py \
# --placeholder_token1='<dog6>' \
# --train_prior_concept1='dog' \
# --prompt='a picture of {} in the snow' \
# --dst_dir='attn_maps/custom_mlm00001' \
# --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_mlm00001_dog6_mprob015_mbatch25_mtarget_masked/checkpoints/learned_embeds_s3000.pt'
