import os
from dataclasses import dataclass, field
from transformers import TrainingArguments as DefaultTrainingArguments
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default="meta-llama/Llama-3.1-8B"
    )
    
    use_fast: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    

@dataclass
class TrainingArguments(DefaultTrainingArguments):

    output_dir: Optional[str] = field(
        default="./pruned_model",
    )

    layer_intervals: Optional[int] = field(default=8)
    
    cosine_num_data: Optional[int] = field(default=50)
    train_num_data: Optional[int] = field(default=100000)
    
    batch_size: Optional[int] = field(default=8)
    
    gradient_accumulation_step: Optional[int] = field(default=16)
    
    epoches: Optional[int] = field(default=1)
    
    lr: float = field(default=2e-4)
    wd: float = field(default=1e-3)

    min_lr: float = field(default=5e-5)


    

    
    

    
    

