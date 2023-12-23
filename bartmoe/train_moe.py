import datasets
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, AutoConfig, BartConfig, BartForConditionalGeneration

from bartmoe import utils
from bart_patch import apply_sliding_window_patch
from bartmoe.configuration_bartmoe import BartMoeConfig
from bartmoe.modeling_bartmoe import BartMOE

from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np

apply_sliding_window_patch()

checkpoint = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
config: BartConfig = AutoConfig.from_pretrained(checkpoint)
config = BartMoeConfig.from_bart_config(config)

model = BartMOE.from_pretrained(
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
eval_dataloader = DataLoader(
    e_tokenized_datasets, batch_size=1, shuffle=True, collate_fn=data_collator
)
datas = data_collator([tokenized_datasets[i] for i in [0, 1, ]])

# Debug
batch = utils.debug_data_processing(train_dataloader)
# model.to("cuda", dtype=torch.bfloat16)
# model.train()
# batch.to("cuda")

utils.freeze_original_bart_decoder_params(model)

nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {nb_trainable_params}")

# Training

optimizer = AdamW(model.parameters(), lr=2e-4)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=1000
)

num_train_epochs = 1
eval_steps = 1000
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = int(num_train_epochs * num_update_steps_per_epoch * 0.1)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

progress_bar = tqdm(range(num_training_steps))

completed_steps = 0

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        outputs = model(**batch)
        loss = outputs.loss
        if step % 100 == 0:
            accelerator.print(
                {
                    "lr": lr_scheduler.get_lr(),
                    "steps": completed_steps,
                    "loss/train": loss.item(),
                }
            )
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        completed_steps += 1
        # Evaluation
        if completed_steps % eval_steps == 0:
            model.eval()
            eval_loss, perplexity = utils.evaluate(model, eval_dataloader, accelerator)
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()




print("saving model")
model.save_pretrained("../final_moe_result")
