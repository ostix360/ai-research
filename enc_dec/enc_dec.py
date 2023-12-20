import datasets
import torch
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, GPT2Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BertTokenizer

import utils
from configuration_enc_dec import EncDecConfig
from data_collator import DataCollatorForSeq2Seq
from modeling_enc_dec import EncDec

enc_model = "bert-base-uncased"
dec_model = "gpt2"
encdec_config = EncDecConfig(enc_model, dec_model)
model = EncDec(encdec_config)
t_dataset = datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[:5000]")
e_dataset = datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[-100:]")

cutoff_len = 1024


enc_tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(enc_model)
dec_tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained(dec_model)
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
        input_ids = torch.tensor(input_ids)
        if length == cutoff_len:    # replace pad_token_id with -100 for labels
            input_batch.append({
                "dec_input_ids": input_ids,
                "dec_attention_mask": input_ids.ne(dec_tokenizer.pad_token_id),
                "labels": input_ids,
                "attention_mask": enc_input_id.ne(enc_tokenizer.pad_token_id),
                "input_ids": enc_input_id,
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
    df = pd.DataFrame(data)
    df['problem'] = '###USER: ' + df['problem']
    df['solution'] = df['problem'] + '\n###ASSISTANT: ' + df['solution']
    return tokenize(df['problem'].tolist(), df['solution'].tolist())


tokenized_datasets = tokenize_func(t_dataset)
e_tokenized_datasets = tokenize_func(e_dataset)

print(f"The column names are: {list(tokenized_datasets.features.keys())}")
data_collator = DataCollatorForSeq2Seq(dec_tokenizer, model=model)
train_dataloader = DataLoader(
    tokenized_datasets, batch_size=1, shuffle=True, collate_fn=data_collator
)
datas = data_collator([tokenized_datasets[i] for i in [0, 1,]])

# Debug
batch = utils.debug_data_processing(train_dataloader)
print(batch)

# model.to("cuda")
# batch.to("cuda")

# out = model(**batch)

training_args = Seq2SeqTrainingArguments(
    output_dir="../result_code",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=1,
    eval_steps=1000,
    bf16=True,
    bf16_full_eval=True,
    save_strategy="steps",
    save_steps=10000,
    warmup_steps=2,
    learning_rate=3e-5,
    # optim="adafactor",
    save_total_limit=1,
    remove_unused_columns=False,

)

trainer = Seq2SeqTrainer(
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
model.save_pretrained("../final_result_code2")
