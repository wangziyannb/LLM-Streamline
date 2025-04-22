import copy
import os
import pickle
from itertools import chain

import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_from_disk, load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from modeling_llama import LlamaModel
from scheduler import get_cosine_schedule_with_warmup


def process_datasets(dataset, train_num_data, tokenizer):
    '''
    We divided the proportions of RedPajamaCommonCrawl, RedPajamaArXiv,
    and RedPajamaBook by a normalization value because the data length
    in these domains is higher than in other domains.
    '''
    proportions = {
        "RedPajamaC4": 0.492,
        "RedPajamaStackExchange": 0.01,
        "RedPajamaCommonCrawl": 0.361 / 3,
        "RedPajamaGithub": 0.008,
        "RedPajamaWikipedia": 0.031,
        "RedPajamaArXiv": 0.007 / 20,
        "RedPajamaBook": 0.091 / 200
    }

    filtered_datasets = {
        name: dataset.filter(lambda x: x['meta'] == {"redpajama_set_name": f"{name}"})
        for name in proportions.keys()
    }

    test_datasets = []
    train_datasets = []

    for name, proportion in proportions.items():
        split = filtered_datasets[name].train_test_split(test_size=(3000 * proportion) / len(filtered_datasets[name]))
        test_datasets.append(split['test'])
        train_split = \
            split['train'].train_test_split(test_size=1 - (train_num_data * proportion) / len(split['train']))['train']
        train_datasets.append(train_split)

    dataset, test_dataset = concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)

    tokenizer.pad_token = tokenizer.eos_token

    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    block_size = 2048

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    test_dataset = test_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    return dataset, test_dataset


if __name__ == '__main__':
    device = 'cuda'
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)

    config = AutoConfig.from_pretrained('meta-llama/Llama-2-7b-hf')
    '''
    Since Llama-3.1-8B has 32 layers, we will prune layers 21 to 30 while keeping layers 31 and 32. 
    The training data should be prepared such that the input to the 20th layer serves as the input to the lightweight layer, 
    and the output of the 30th layer serves as the output of the lightweight layer. 
    Therefore, during the training of the lightweight layer, layers 31 and 32 are not involved.
    '''
    config.num_hidden_layers = 2
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaModel(config)
    llama_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

    model_dict = model.state_dict()
    llama_dict = llama_model.state_dict()

    model_dict['embed_tokens.weight'] = llama_dict['model.embed_tokens.weight']
    for i in range(2):
        model_dict['layers.{}.self_attn.q_proj.weight'.format(i)] = llama_dict[
            'model.layers.{}.self_attn.q_proj.weight'.format(i)]
        model_dict['layers.{}.self_attn.k_proj.weight'.format(i)] = llama_dict[
            'model.layers.{}.self_attn.k_proj.weight'.format(i)]
        model_dict['layers.{}.self_attn.v_proj.weight'.format(i)] = llama_dict[
            'model.layers.{}.self_attn.v_proj.weight'.format(i)]
        model_dict['layers.{}.self_attn.o_proj.weight'.format(i)] = llama_dict[
            'model.layers.{}.self_attn.o_proj.weight'.format(i)]

        model_dict['layers.{}.mlp.gate_proj.weight'.format(i)] = llama_dict[
            'model.layers.{}.mlp.gate_proj.weight'.format(i)]
        model_dict['layers.{}.mlp.up_proj.weight'.format(i)] = llama_dict[
            'model.layers.{}.mlp.up_proj.weight'.format(i)]
        model_dict['layers.{}.mlp.down_proj.weight'.format(i)] = llama_dict[
            'model.layers.{}.mlp.down_proj.weight'.format(i)]

        model_dict['layers.{}.input_layernorm.weight'.format(i)] = llama_dict[
            'model.layers.{}.input_layernorm.weight'.format(i)]
        model_dict['layers.{}.post_attention_layernorm.weight'.format(i)] = llama_dict[
            'model.layers.{}.post_attention_layernorm.weight'.format(i)]

    model.load_state_dict(model_dict)
    del llama_model
    model = model.to(device)
    for name, p in model.named_parameters():
        if "replace_layer" in name:
            continue
        else:
            if p.requires_grad == True:
                p.requires_grad = False

    if os.path.exists("slimpajama-Llama-2-tokenized-0.06b"):
        datasets = load_from_disk("slimpajama-Llama-2-tokenized-0.06b")
        dataset = datasets['train']
        test_dataset = datasets['validation']
    else:
        dataset = load_dataset('DKYoon/SlimPajama-6B')['train']
        dataset, test_dataset = process_datasets(dataset, 100000, tokenizer)

        # dataset = load_from_disk('/data/slimpajama-0.5B-Llama-3-tokenized')
        # eval_dataset = dataset['validation']
        datasets = DatasetDict({'validation': test_dataset, 'train': dataset})
        datasets.save_to_disk("slimpajama-Llama-2-tokenized-0.06b")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator,
                                 shuffle=True)
    train_dataloader = DataLoader(dataset, batch_size=32, collate_fn=data_collator,
                                  shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.95))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * 0.03,
        num_training_steps=len(train_dataloader),
        max_learning_rate=1e-3,
        min_learning_rate=2.5e-5,
    )

    train_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, test_dataloader, model, optimizer
    )

    mse_loss = nn.MSELoss()

    best_loss = 10000
    for epoch in range(20):
        model.train()
        for step, batch in enumerate(
                tqdm(train_dataloader, desc=f"Epoch {epoch}", total=len(train_dataloader))
        ):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                output_dict = outputs.last_hidden_state[-1]
                labels = output_dict["target_output"]
                outputs = output_dict["replace_layer_output"]
                loss = mse_loss(labels, outputs)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 500 == 0:
                model.eval()
                losses = []
                for step, batch in tqdm(enumerate(test_dataloader)):
                    with torch.no_grad():
                        input_ids = batch['input_ids']
                        attention_mask = batch['attention_mask']
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        output_dict = outputs.last_hidden_state[-1]
                        labels = output_dict["target_output"]
                        outputs = output_dict["replace_layer_output"]

                    loss = mse_loss(labels, outputs)
                    losses.append(accelerator.gather_for_metrics(loss.repeat(64)))

                losses = torch.cat(losses)
                eval_loss = torch.mean(losses)
                print(eval_loss)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    # model.save_pretrained('')
                model.train()
    # torch.save(model, 'model.bin')
    torch.save(
        {'config': copy.deepcopy(model.replace_layer.config), 'u_pruned': copy.deepcopy(model.replace_layer.up_proj),
         'g_pruned': copy.deepcopy(model.replace_layer.gate_proj),
         'd_pruned': copy.deepcopy(model.replace_layer.down_proj)}, 'sub_mlp.pth')

    # model.save_pretrained('')
