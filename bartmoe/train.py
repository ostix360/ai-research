import datasets
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, AutoConfig, BartConfig, BartForConditionalGeneration

import utils
from bart_patch import apply_sliding_window_patch

apply_sliding_window_patch()

checkpoint = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
config: BartConfig = AutoConfig.from_pretrained(checkpoint)

config.sliding_window = 1024
model = BartForConditionalGeneration.from_pretrained(
    checkpoint,
    config=config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)

t_dataset = datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")
e_dataset = datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[-100:]")

cutoff_len = 1024


def tokenize(user, assistant):
    model_inputs = tokenizer(user, text_target=assistant, truncation=True, max_length=cutoff_len)
    return Dataset.from_dict(model_inputs)


def tokenize_func(data):
    df = pd.DataFrame(data)
    df['problem'] = '###USER: ' + df['problem']
    df['solution'] = '###ASSISTANT: ' + df['solution']
    return tokenize(df['problem'].to_list(), df['solution'].to_list())


tokenized_datasets = tokenize_func(t_dataset['train'])
e_tokenized_datasets = tokenize_func(e_dataset)

print(f"The column names are: {list(tokenized_datasets.features.keys())}")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
train_dataloader = DataLoader(
    tokenized_datasets, batch_size=1, shuffle=True, collate_fn=data_collator
)
datas = data_collator([tokenized_datasets[i] for i in [0, 1,]])

# Debug
batch = utils.debug_data_processing(train_dataloader)
# model.to("cuda", dtype=torch.bfloat16)
# model.train()
# batch.to("cuda")

training_args = Seq2SeqTrainingArguments(
    output_dir="../result_code",
    per_device_train_batch_size=3,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=25,
    num_train_epochs=2,
    eval_steps=500,
    bf16=True,
    bf16_full_eval=True,
    save_strategy="steps",
    save_steps=10000,
    warmup_steps=2,
    learning_rate=4e-5,
    # optim="adafactor",
    save_total_limit=1,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=e_tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {nb_trainable_params}")

trainer.train()
print("saving model")
model.save_pretrained("../final_result_code")
