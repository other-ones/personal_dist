from transformers import T5Tokenizer, T5ForConditionalGeneration
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
parser.add_argument('--include_special',action='store_true')
args=parser.parse_args()
os.makedirs(args.dst_dir,exist_ok=True)
pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5'
# tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
text_encoder = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
# text_encoder = CLIPTextModel.from_pretrained(
#         pretrained_model_name_or_path,
#         subfolder="text_encoder",
#         revision=None,
#     )


token_embeds = text_encoder.get_input_embeddings().weight.data
prompt=args.prompt.format(f'{args.placeholder_token1} {args.train_prior_concept1}')
inputs=tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
print(tokenizer.eos_token_id,'eos_token_id')
print(tokenizer.pad_token_id,'pad_token_id')
print(tokenizer.bos_token_id,'bos_token_id')
input_ids = inputs['input_ids']  # Tensor of token IDs
non_pad=input_ids!=tokenizer.pad_token_id
non_eos=input_ids!=tokenizer.eos_token_id
first_eos=torch.argmin(non_eos.float())
nonspecial=torch.logical_and(non_pad,non_eos)
print(non_pad.shape,'non_pad.shape')
print(non_eos.shape,'non_eos.shape')
print(nonspecial.shape,'nonspecial.shape',torch.sum(nonspecial))
print(torch.sum(nonspecial),'nonspecial')
# Pass the inputs through the model
# outputs = text_encoder(inputs,output_attentions=True)
outputs=text_encoder.encoder(input_ids,output_hidden_states=True,output_attentions=True)
attention = outputs.attentions  # Shape: (layers, batch, heads, tokens, tokens)
# attention: layers,batch,heads,tokens,tokens
# for item in attention:
#     print(item.shape,'item.shape')  # batch,heads,tokens,tokens


# Select a specific layer
layer = 0  # Change this to visualize different layers
# if args.include_special:
# print(first_eos,'first_eos')
# print(nonspecial,'nonspecial')
nonspecial=nonspecial[0]
for layer in range(len(attention)):
    attn_weights = attention[layer][0]  # Shape: (heads, tokens, tokens)
    avg_attn_weights = torch.mean(attn_weights, dim=0).detach().cpu() # Shape: (tokens, tokens)
    print(input_ids.shape,'input_ids.shape')
    print(attn_weights.shape,'attn_weights.shape') #heads,tokens,tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # tokens=[item.replace('▁','') for item in tokens]
    print(tokens,'tokens')
    plt.figure(figsize=(8, 8))
    print(avg_attn_weights.shape,'avg_attn_weights.shape1')
    avg_attn_weights=avg_attn_weights[nonspecial][:,nonspecial]
    avg_attn_weights=avg_attn_weights.numpy()
    print(avg_attn_weights.shape,'avg_attn_weights.shape2')
    im = plt.imshow(avg_attn_weights, cmap="viridis")
    tokens=np.array(tokens)[nonspecial]
    tokens=tokens.tolist()
    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
    plt.yticks(ticks=range(len(tokens)), labels=tokens)
    plt.colorbar(im)
    plt.title(f"Average Attention Map for Layer {layer + 1}")
    plt.tight_layout()
    plt.savefig("{}/average_attention_map_layer{}.png".format(args.dst_dir,layer+1))
    # plt.show()