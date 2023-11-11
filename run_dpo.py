# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from typing import Dict
from trl import DPOTrainer
from trl.trainer import ConstantLengthDataset


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(args):
    dataset_name = args.dataset_name
    split = args.split
    sanity_check = False
    silent = False
    cache_dir = None

    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)


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
        per_device_train_batch_size=4,
        max_steps=500,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,  # match results in blog post
        eval_steps=500,
        output_dir=save_dir,
        optim="rmsprop",
        warmup_steps=150,
        report_to="none",
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    data_args = {"dataset_name": "Anthropic/hh-rlhf",
                 "subset": "data/finetune",
                 "split": "train",
                 "size_valid_set": 4000,
                 "streaming": False,
                 "seq_length": 512,
                 "num_workers": 4,
                 "shuffle_buffer": 5000}

    data_args = DictObj(data_args)

    train_dataset = get_hh(data_args)

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(save_dir)

    output_dir = os.path.join(save_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()