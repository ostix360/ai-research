import datasets
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, AutoConfig, BartConfig, BartForConditionalGeneration

from bartmoe import utils
from bartmoe.bart_patch import apply_sliding_window_patch
from bartmoe.configuration_bartmoe import BartMoeConfig
from bartmoe.modeling_bartmoe import BartMOE

from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np

apply_sliding_window_patch() # 500 sliding window

checkpoint = "./bart_moe_labeled"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
config: BartMoeConfig = AutoConfig.from_pretrained(checkpoint)
config.sliding_window = 550
config.expert_num_layers = 12
config.expert_num_heads = 16
config.dropout = 0.5
config.max_length = 512

model = BartMOE.from_pretrained(
    checkpoint,
    config=config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map=0,
)

t_code_dataset = datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[:75000]")
e_code_dataset = datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[-20:]")

t_metamath_dataset = datasets.load_dataset("meta-math/MetaMathQA", split="train[:250000]")
e_metamath_dataset = datasets.load_dataset("meta-math/MetaMathQA", split="train[-20:]")


cutoff_len = 1024


def tokenize(user, assistant, ds_label):
    model_inputs = tokenizer(user, text_target=assistant, truncation=True, max_length=cutoff_len)
    if ds_label is not None:
        model_inputs['encoder_labels'] = [ds_label for _ in range(len(model_inputs['input_ids']))]
    model_inputs['expert_train'] = [2 for _ in range(len(model_inputs['input_ids']))]
    return Dataset.from_dict(model_inputs)


def tokenize_func(data, user_key, assistant_key, ds_label):
    df = pd.DataFrame(data)
    df[user_key] = '###USER: ' + df[user_key]
    df[assistant_key] = '###ASSISTANT: ' + df[assistant_key]
    return tokenize(df[user_key].to_list(), df[assistant_key].to_list(), ds_label)


# t_code_tokenized_datasets = tokenize_func(t_code_dataset, 'problem', 'solution', None)
# e_code_tokenized_datasets = tokenize_func(e_code_dataset, 'problem', 'solution', 1)

t_metamath_tokenized_datasets = tokenize_func(t_metamath_dataset, 'query', 'response', None)
e_metamath_tokenized_datasets = tokenize_func(e_metamath_dataset, 'query', 'response', 2)


tokenized_datasets = concatenate_datasets([t_metamath_tokenized_datasets,])
e_tokenized_datasets = concatenate_datasets([e_metamath_tokenized_datasets,])

print(f"The column names are: {list(tokenized_datasets.features.keys())}")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, max_length=cutoff_len)
train_dataloader = DataLoader(
    tokenized_datasets, batch_size=5, shuffle=False, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    e_tokenized_datasets, batch_size=20, shuffle=True, collate_fn=data_collator
)
datas = data_collator([tokenized_datasets[i] for i in [0, 1, ]])

# Debug
batch = utils.debug_data_processing(train_dataloader)
model.train()
# batch.to("cuda")
# for _ in range(40):
#     out = model(**batch)

utils.freeze_original_bart_params(model)
utils.freeze_expert_params(model, 0)
utils.freeze_expert_params(model, 1)
# utils.freeze_expert_params(model, 2)
utils.freeze_expert_params(model, 3)

nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {nb_trainable_params}")

# Training

training_args = Seq2SeqTrainingArguments(
    output_dir="./bart_moe_experts",
    per_device_train_batch_size=3,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=1,
    eval_steps=500,
    bf16=True,
    bf16_full_eval=True,
    save_strategy="steps",
    save_steps=10000,
    warmup_steps=2,
    learning_rate=1e-4,
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
    compute_metrics=utils.compute_metrics
)
trainer.evaluate()
trainer.train()
trainer.evaluate()


#
# num_train_epochs = 1
# eval_steps = 500
# num_update_steps_per_epoch = len(train_dataloader)
# num_training_steps = int(num_train_epochs * num_update_steps_per_epoch)
#
# optimizer = AdamW(model.parameters(), lr=1e-5)
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=20,
#     num_training_steps=num_training_steps
# )
#
# accelerator = Accelerator()
# model.to("cuda", dtype=torch.bfloat16)
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )
#
# progress_bar = tqdm(range(num_training_steps))
#
#
# def train(num_training_steps=num_training_steps, eval_steps=eval_steps):
#     print(f"Start training with {num_training_steps} steps, {eval_steps} steps per evaluation epoch.")
#     completed_steps = 0
#     for epoch in range(num_train_epochs):
#         # Training
#         model.train()
#         for step, batch in tqdm(
#                 enumerate(train_dataloader, start=1), total=num_training_steps,
#         ):
#             try:
#                 outputs = model(**batch)
#             except RuntimeError as e:
#                 if "device-side assert triggered" in str(e):
#                     print(f"Error occurred at step {step}")
#                     print(f"Batch data: {batch}")
#                     break
#             loss = outputs.loss
#             encoder_loss = outputs.encoder_loss
#             if step % 50 == 0:
#                 accelerator.print(
#                     {
#                     "lr": lr_scheduler.get_last_lr()[0],
#                     "steps": completed_steps,
#                     "loss/train": loss.item(),
#                     # "encoder_loss/train": encoder_loss.item(),
#                 }
#             )
#             if loss is not None:
#                 accelerator.backward(loss)
#             if encoder_loss is not None:
#                 accelerator.backward(encoder_loss)
#
#             optimizer.step()    # doesn't work
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             progress_bar.update(1)
#             completed_steps += 1
#             # Evaluation
#             if completed_steps % eval_steps == 0:
#                 model.eval()
#                 eval_loss, perplexity = utils.evaluate(model, eval_dataloader, accelerator)
#                 accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
#                 eval_encoder_loss, eval_perplexity = utils.evaluate_classification(model, eval_dataloader, accelerator)
#                 accelerator.print({"encoder_loss/eval": eval_encoder_loss, "eval_perplexity": eval_perplexity})
#                 model.train()

# train()
print("saving model")
model.save_pretrained("./bart_moe_experts")
tokenizer.save_pretrained("./bart_moe_experts")
