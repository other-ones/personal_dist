import os
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import numpy as np
from transformers import CLIPTextModel,CLIPTokenizer, T5ForConditionalGeneration
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--placeholder_token1')
parser.add_argument('--learned_embed_path1')
parser.add_argument('--train_prior_concept1')
parser.add_argument('--prompt')
parser.add_argument('--dst_dir')
parser.add_argument('--distill',type=float)
parser.add_argument('--include_special',action='store_true')
parser.add_argument('--include_all',action='store_true')
args=parser.parse_args()
os.makedirs(args.dst_dir,exist_ok=True)
pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5'
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=None,
    )
placeholder_tokens = [args.placeholder_token1]
tokenizer.add_tokens(placeholder_tokens)
placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_tokens)
text_encoder.resize_token_embeddings(len(tokenizer))
token_embeds = text_encoder.get_input_embeddings().weight.data
learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token1]
learned_embed1=learned_embed1[args.placeholder_token1]
with torch.no_grad():
    token_embeds[placeholder_token_id1] = learned_embed1 #
    # token_embeds[placeholder_token_id2] = learned_embed2 #token_embeds[initializer_token_id].clone()
# Tokenize the input
# text = "a picture of a cute dog in the snow"
prompt=args.prompt.format(f'{args.placeholder_token1} {args.train_prior_concept1}')
# inputs = tokenizer(text, return_tensors="pt")
inputs=tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
print(tokenizer.eos_token_id,'eos_token_id')
print(tokenizer.bos_token_id,'bos_token_id')
input_ids = inputs['input_ids']  # Tensor of token IDs
is_keyword_tokens_list=input_ids==placeholder_token_id1[0]
assert torch.sum(is_keyword_tokens_list)==len(is_keyword_tokens_list)

eos_idxs=(input_ids==tokenizer.eos_token_id).float().argmax(1)
eos_idxs=torch.clip(eos_idxs,None,(tokenizer.model_max_length-1))
non_bos=input_ids!=tokenizer.bos_token_id
non_eos=input_ids!=tokenizer.eos_token_id
first_eos=torch.argmin(non_eos.float())
nonspecial=torch.logical_and(non_bos,non_eos).reshape(-1)
# Pass the inputs through the model


print(input_ids.shape,'input_ids.shape')
with torch.no_grad():
    outputs = text_encoder(**inputs,output_attentions=True,
                        distill=args.distill,
                        eos_tokens_list=eos_idxs,
                        is_keyword_tokens1=is_keyword_tokens_list,
    )
    attention = outputs.attentions  # Shape: (layers, batch, heads, tokens, tokens)

    # Select a specific layer
    layer = 0  # Change this to visualize different layers
    if args.include_special or args.include_all:
        nonspecial[0]=True
        nonspecial[first_eos]=True

    for layer in range(12):
        attn_weights = attention[layer][0]  # Shape: (heads, tokens, tokens)

        avg_attn_weights = torch.mean(attn_weights, dim=0).detach().cpu() # Shape: (tokens, tokens)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens=[item.replace('</w>','')for item in tokens]
        plt.figure(figsize=(15, 15))
        print(avg_attn_weights.shape,'avg_attn_weights.shape1')
        if not args.include_all:
            avg_attn_weights=avg_attn_weights[nonspecial][:,nonspecial]
            # avg_attn_weights=avg_attn_weights[nonspecial][:,nonspecial]
            tokens=np.array(tokens)[nonspecial]

        avg_attn_weights=avg_attn_weights.numpy()
        print(avg_attn_weights.shape,'avg_attn_weights.shape2')
        im = plt.imshow(avg_attn_weights, cmap="viridis")
        tokens=np.array(tokens)
        tokens=tokens.tolist()
        plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
        plt.yticks(ticks=range(len(tokens)), labels=tokens)
        plt.colorbar(im)
        plt.title(f"Average Attention Map for Layer {layer + 1}")
        plt.tight_layout()
        plt.savefig("{}/average_attention_map_layer{}.png".format(args.dst_dir,layer+1))
        # plt.show()