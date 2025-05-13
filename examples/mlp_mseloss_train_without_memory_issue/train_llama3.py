import copy
import os
import pickle
from itertools import chain

import torch
import torch.nn as nn
import wandb  # Weights & Biases for experiment tracking
from accelerate import Accelerator
from datasets import load_from_disk, load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from modeling_llama import LlamaModel
from scheduler import get_cosine_schedule_with_warmup


def process_datasets(dataset, train_num_data, tokenizer):
    """
    We divide the proportions of RedPajama datasets to balance domain representation.
    """
    proportions = {
        "RedPajamaC4": 0.492,
        "RedPajamaStackExchange": 0.01,
        "RedPajamaCommonCrawl": 0.361 / 3,
        "RedPajamaGithub": 0.008,
        "RedPajamaWikipedia": 0.031,
        "RedPajamaArXiv": 0.007 / 20,
        "RedPajamaBook": 0.091 / 200
    }

    # Filter by sub-dataset
    filtered_datasets = {
        name: dataset.filter(lambda x: x['meta'] == {"redpajama_set_name": f"{name}"})
        for name in proportions.keys()
    }

    test_datasets, train_datasets = [], []
    for name, prop in proportions.items():
        split = filtered_datasets[name].train_test_split(
            test_size=(3000 * prop) / len(filtered_datasets[name])
        )
        test_datasets.append(split['test'])
        train_split = split['train'].train_test_split(
            test_size=1 - (train_num_data * prop) / len(split['train'])
        )['train']
        train_datasets.append(train_split)

    dataset = concatenate_datasets(train_datasets)
    test_dataset = concatenate_datasets(test_datasets)

    tokenizer.pad_token = tokenizer.eos_token
    column_names = dataset.column_names
    text_col = "text" if "text" in column_names else column_names[0]

    def tokenize_fn(examples):
        return tokenizer(examples[text_col])

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=column_names)
    test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=column_names)

    block_size = wandb.config.block_size

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_len = len(concatenated[list(examples.keys())[0]])
        total_len = (total_len // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(group_texts, batched=True)
    test_dataset = test_dataset.map(group_texts, batched=True)
    return dataset, test_dataset


if __name__ == '__main__':
    # Initialize Accelerator
    device = 'cuda'
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)

    # Initialize Weights & Biases
    wandb.init(
        project="slimpajama_prune_llama3",
        config={
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "betas": (0.9, 0.95),
            "block_size": 2048,
            "model_name": "meta-llama/Meta-Llama-3-8B"
        }
    )
    config = wandb.config

    # Model and tokenizer setup
    auto_config = AutoConfig.from_pretrained(config.model_name)
    auto_config.num_hidden_layers = 3  # only keep layers 21-30 for replacement layer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Custom lightweight LlamaModel
    model = LlamaModel(auto_config)
    llama_model = AutoModelForCausalLM.from_pretrained(config.model_name)

    # Load pretrained weights into the replace_layer model
    state_dict = model.state_dict()
    llama_dict = llama_model.state_dict()
    state_dict['embed_tokens.weight'] = llama_dict['model.embed_tokens.weight']
    for i in range(auto_config.num_hidden_layers):
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            state_dict[f'layers.{i}.self_attn.{proj}.weight'] = llama_dict[f'model.layers.{i}.self_attn.{proj}.weight']
        for mlp_proj in ['gate_proj', 'up_proj', 'down_proj']:
            state_dict[f'layers.{i}.mlp.{mlp_proj}.weight'] = llama_dict[f'model.layers.{i}.mlp.{mlp_proj}.weight']
        for ln in ['input_layernorm', 'post_attention_layernorm']:
            state_dict[f'layers.{i}.{ln}.weight'] = llama_dict[f'model.layers.{i}.{ln}.weight']
    model.load_state_dict(state_dict)
    del llama_model

    model = model.to(device)
    # Freeze all except replace_layer
    for name, p in model.named_parameters():
        if 'replace_layer' not in name:
            p.requires_grad = False

    # Dataset loading or processing
    if os.path.exists("slimpajama-Llama-3-tokenized-0.06b"):
        datasets = load_from_disk("slimpajama-Llama-3-tokenized-0.06b")
        train_dataset = datasets['train']
        test_dataset = datasets['validation']
    else:
        raw = load_dataset('DKYoon/SlimPajama-6B')['train']
        train_dataset, test_dataset = process_datasets(raw, 100000, tokenizer)
        ds = DatasetDict({'train': train_dataset, 'validation': test_dataset})
        ds.save_to_disk("slimpajama-Llama-3-tokenized-0.06b")

    # DataLoaders
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collator, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collator, shuffle=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(len(train_loader) * 0.03),
        num_training_steps=len(train_loader) * config.epochs,
        max_learning_rate=config.learning_rate,
        min_learning_rate=2.5e-5,
    )

    # Prepare with accelerator
    train_loader, test_loader, model, optimizer = accelerator.prepare(
        train_loader, test_loader, model, optimizer
    )

    # Track gradients and parameters
    wandb.watch(model, log=None, log_freq=10)

    mse_loss = nn.MSELoss()
    best_loss = float('inf')
    global_step = 0

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                output_dict = outputs.last_hidden_state[-1]
                labels = output_dict["target_output"]
                preds = output_dict["replace_layer_output"]
                loss = mse_loss(labels, preds)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            # Log training metrics
            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch,
                "train/step": global_step,
                "lr": lr_scheduler.get_last_lr()[0]
            }, step=global_step)

            # Periodic evaluation
            if global_step % 300 == 0:
                model.eval()
                eval_losses = []
                for _, eval_batch in enumerate(test_loader):
                    with torch.no_grad():
                        input_ids = batch['input_ids']
                        attention_mask = batch['attention_mask']
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                        od = out.last_hidden_state[-1]
                        lbls = od["target_output"]
                        pr = od["replace_layer_output"]
                    l = mse_loss(lbls, pr)
                    eval_losses.append(accelerator.gather_for_metrics(loss.repeat(32)))
                losses = torch.cat(eval_losses)
                eval_loss = torch.mean(losses)
                wandb.log({"eval/loss": eval_loss, "eval/global_step": global_step}, step=global_step)

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    # Save best replace_layer weights
                    torch.save({
                        'config': copy.deepcopy(model.replace_layer.config),
                        'u_pruned': copy.deepcopy(model.replace_layer.up_proj),
                        'g_pruned': copy.deepcopy(model.replace_layer.gate_proj),
                        'd_pruned': copy.deepcopy(model.replace_layer.down_proj)
                    }, 'sub_mlp.pth')
                    # torch.save({
                        # 'config': copy.deepcopy(model.config),
                        # 'state_dict': model.replace_layer.state_dict(),
                        # 'u_pruned': copy.deepcopy(model.replace_layer.up_proj),
                        # 'g_pruned': copy.deepcopy(model.replace_layer.gate_proj),
                        # 'd_pruned': copy.deepcopy(model.replace_layer.down_proj)
                    # }, 'sub_decoder_layer_2.pth')
                    # wandb.save('sub_mlp.pth')
                model.train()

    # Finish run
    wandb.finish()
