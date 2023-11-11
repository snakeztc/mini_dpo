# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from trl import SFTTrainer
from trl.import_utils import is_xpu_available
from trl.trainer import ConstantLengthDataset


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    #text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    text = example["text"]
    return text


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        # data_dir=args.subset,
        split=args.split,
        use_auth_token=False,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

def main():
    model_name = "../models/gpt2"
    save_dir = "runs"
    training_args = TrainingArguments(
        output_dir=save_dir,
        max_steps=500,
        logging_steps=10,
        save_steps=100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        group_by_length=False,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.05,
        remove_unused_columns=False,
        run_name="sft_llama2",
        use_mps_device=True,
        report_to="none"
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    data_args = {"dataset_name": "timdettmers/openassistant-guanaco",
                 "subset": "data/finetune",
                 "split": "train",
                 "size_valid_set": 4000,
                 "streaming": False,
                 "seq_length": 512,
                 "num_workers": 4,
                 "shuffle_buffer": 5000}

    data_args = DictObj(data_args)

    train_dataset, eval_dataset = create_datasets(tokenizer, data_args)

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=True,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(save_dir)

    output_dir = os.path.join(save_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()