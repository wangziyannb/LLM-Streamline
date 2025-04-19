from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from accelerate import Accelerator
from accelerate.utils import set_seed
from scheduler import get_cosine_schedule_with_warmup
from tqdm import tqdm

from modeling_llama import LlamaModel
import deepspeed

accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=2)

config = AutoConfig.from_pretrained('/fs/fast/u2021000902/cxd/model/Llama-3.1-8B')
'''
Since Llama-3.1-8B has 32 layers, we will prune layers 21 to 30 while keeping layers 31 and 32. 
The training data should be prepared such that the input to the 20th layer serves as the input to the lightweight layer, 
and the output of the 30th layer serves as the output of the lightweight layer. 
Therefore, during the training of the lightweight layer, layers 31 and 32 are not involved.
'''
config.num_hidden_layers = 30  
tokenizer = AutoTokenizer.from_pretrained('/fs/fast/u2021000902/cxd/model/Llama-3.1-8B')
tokenizer.pad_token = tokenizer.eos_token

model = LlamaModel(config)
llama_model = AutoModelForCausalLM.from_pretrained('/fs/fast/u2021000902/cxd/model/Llama-3.1-8B')

model_dict = model.state_dict()
llama_dict = llama_model.state_dict()

model_dict['embed_tokens.weight'] = llama_dict['model.embed_tokens.weight']
for i in range(30):
    model_dict['layers.{}.self_attn.q_proj.weight'.format(i)] = llama_dict['model.layers.{}.self_attn.q_proj.weight'.format(i)]
    model_dict['layers.{}.self_attn.k_proj.weight'.format(i)] = llama_dict['model.layers.{}.self_attn.k_proj.weight'.format(i)]
    model_dict['layers.{}.self_attn.v_proj.weight'.format(i)] = llama_dict['model.layers.{}.self_attn.v_proj.weight'.format(i)]
    model_dict['layers.{}.self_attn.o_proj.weight'.format(i)] = llama_dict['model.layers.{}.self_attn.o_proj.weight'.format(i)]

    model_dict['layers.{}.mlp.gate_proj.weight'.format(i)] = llama_dict['model.layers.{}.mlp.gate_proj.weight'.format(i)]
    model_dict['layers.{}.mlp.up_proj.weight'.format(i)] = llama_dict['model.layers.{}.mlp.up_proj.weight'.format(i)]
    model_dict['layers.{}.mlp.down_proj.weight'.format(i)] = llama_dict['model.layers.{}.mlp.down_proj.weight'.format(i)]
    
    model_dict['layers.{}.input_layernorm.weight'.format(i)] = llama_dict['model.layers.{}.input_layernorm.weight'.format(i)]
    model_dict['layers.{}.post_attention_layernorm.weight'.format(i)] = llama_dict['model.layers.{}.post_attention_layernorm.weight'.format(i)]
        

model_dict['replace_layer.self_attn.q_proj.weight'] = llama_dict['model.layers.{}.self_attn.q_proj.weight'.format(19)]  #We use the weights from the 20th layer to initialize the lightweight layer.
model_dict['replace_layer.self_attn.k_proj.weight'] = llama_dict['model.layers.{}.self_attn.k_proj.weight'.format(19)]
model_dict['replace_layer.self_attn.v_proj.weight'] = llama_dict['model.layers.{}.self_attn.v_proj.weight'.format(19)]
model_dict['replace_layer.self_attn.o_proj.weight'] = llama_dict['model.layers.{}.self_attn.o_proj.weight'.format(19)]
model_dict['replace_layer.mlp.gate_proj.weight'] = llama_dict['model.layers.{}.mlp.gate_proj.weight'.format(19)]
model_dict['replace_layer.mlp.up_proj.weight'] = llama_dict['model.layers.{}.mlp.up_proj.weight'.format(19)]
model_dict['replace_layer.mlp.down_proj.weight'] = llama_dict['model.layers.{}.mlp.down_proj.weight'.format(19)]
model_dict['replace_layer.input_layernorm.weight'] = llama_dict['model.layers.{}.input_layernorm.weight'.format(19)]
model_dict['replace_layer.post_attention_layernorm.weight'] = llama_dict['model.layers.{}.post_attention_layernorm.weight'.format(19)]

model.load_state_dict(model_dict)
del llama_model

for name,p in model.named_parameters():
    if "replace_layer" in name:
        continue
    else:
        if p.requires_grad == True:
            p.requires_grad = False

dataset = load_from_disk('/fs/fast/u2021000902/cxd/data/SlimPajama6B-llama3-processed')
eval_dataset = dataset['validation']

dataset = dataset['train'].train_test_split(test_size =  300000 / len(dataset['train']))['test']

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(dataset, shuffle=True, collate_fn=data_collator, batch_size=32)
eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=64)

optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-4, weight_decay = 1e-3, betas=(0.9, 0.95))
lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader)*0.03,
        num_training_steps=len(train_dataloader),
        max_learning_rate=2e-4,
        min_learning_rate=5e-6,
    )

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

mse_loss = nn.MSELoss()

best_loss = 10000
for epoch in range(1):
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader)):
        with accelerator.accumulate(model):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids = input_ids, attention_mask=attention_mask)
            labels = outputs.last_hidden_state[0]
            outputs = outputs.last_hidden_state[1]
            loss = mse_loss(labels, outputs)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if (step+1) % 500 == 0:            
            model.eval()
            losses = []
            for step, batch in tqdm(enumerate(eval_dataloader)):
                with torch.no_grad():
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    outputs = model(input_ids = input_ids, attention_mask=attention_mask)
                    labels = outputs.last_hidden_state[0]
                    outputs = outputs.last_hidden_state[1]

                loss = mse_loss(labels, outputs)
                losses.append(accelerator.gather_for_metrics(loss.repeat(64)))

            losses = torch.cat(losses)
            eval_loss = torch.mean(losses)
            print(eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                #model.save_pretrained('')
            model.train()

#model.save_pretrained('')
