import os
from typing import Union, Optional, Callable

import datasets
import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, AutoConfig, PreTrainedModel, PretrainedConfig, \
    BertTokenizer
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from transformers.models.bert.modeling_bert import BertModel

import utils
from enc_dec.configuration_enc_dec import EncDecConfig
from enc_dec.modeling_enc_dec import EncDec

enc_model = "bert-base-uncased"
dec_model = "gpt2"
encdec_config = EncDecConfig(enc_model, dec_model)
model = EncDec(encdec_config)
model.save_pretrained("./result_code")
t_dataset = datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[:1000]")
e_dataset = datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[-10:]")

cutoff_len = 1024


enc_tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(enc_model)
dec_tokenizer = AutoTokenizer.from_pretrained(dec_model)
dec_tokenizer.pad_token = dec_tokenizer.eos_token


def tokenize(user, assistant):
    enc_input_ids = enc_tokenizer(user, truncation=True, max_length=512)["input_ids"]

    enc_input_ids = [enc_input_id + [enc_tokenizer.pad_token_id] * (512 - len(enc_input_id)) for enc_input_id in enc_input_ids]

    dec_input_ids = dec_tokenizer(assistant, padding="max_length", truncation=True, max_length=cutoff_len,
                                  return_overflowing_tokens=True, return_length=True,)

    input_batch = []
    i = 0
    for length, input_ids in zip(dec_input_ids["length"], dec_input_ids["input_ids"]):
        overflow_to_sample_mapping = dec_input_ids["overflow_to_sample_mapping"][i]
        enc_input_id = enc_input_ids[overflow_to_sample_mapping]
        enc_input_id = torch.tensor(enc_input_id)
        labels = [1] * len(input_ids)
        input_ids = torch.tensor(input_ids)
        if length == cutoff_len:
            input_batch.append({
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(dec_tokenizer.pad_token_id),
                "labels": labels,
                "enc_attention_mask": enc_input_id.ne(enc_tokenizer.pad_token_id),
                "enc_input_ids": enc_input_id,
            })
        i += 1
    return Dataset.from_list(input_batch)


# def tokenize(prompt):
#     input_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=cutoff_len)["input_ids"]
#     input_ids = torch.tensor(input_ids)
#     labels = [1] * len(input_ids)
#     return {
#         "input_ids": input_ids,
#         "labels": labels,
#         "attention_mask": input_ids.ne(tokenizer.pad_token_id),
#     }

def tokenize_func(data):
    return tokenize([f"###USER: {d}\n" for d in data["problem"]], [f"###ASSISTANT: {d}" for d in data["solution"]])


tokenized_datasets = tokenize_func(t_dataset)
e_tokenized_datasets = tokenize_func(e_dataset)

print(f"The column names are: {list(tokenized_datasets.features.keys())}")
data_collator = DataCollatorForLanguageModeling(dec_tokenizer, mlm=False)
train_dataloader = DataLoader(
    tokenized_datasets, batch_size=1, shuffle=True, collate_fn=data_collator
)

# Debug
batch = utils.debug_data_processing(train_dataloader)

out = model(**batch)

training_args = TrainingArguments(
    output_dir="../result_code",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=25,
    num_train_epochs=1,
    eval_steps=500,
    bf16=True,
    bf16_full_eval=True,
    save_strategy="steps",
    save_steps=500,
    warmup_steps=2,
    learning_rate=2e-5,
    save_total_limit=1,
    remove_unused_columns=False,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=e_tokenized_datasets,
    tokenizer=dec_tokenizer,
    data_collator=data_collator,
)

nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {nb_trainable_params}")

trainer.train()
print("hi")