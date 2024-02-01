import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastMistralModel
import fff_mistral_patch

fff_mistral_patch.patch_to_unsloth_mistral()


model_name = "unsloth/mistral-7b-bnb-4bit"
max_seq_length = 2048
dtype = None
load_in_4bit = True

t_code_dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[:100]")
e_code_dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[-1:]")

t_metamath_dataset = load_dataset("meta-math/MetaMathQA", split="train[:100]")
e_metamath_dataset = load_dataset("meta-math/MetaMathQA", split="train[-1:]")

# t_code2_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train[:370]")
# e_code2_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train[-20:]")

t_ultra_chat_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:250]")
e_ultra_chat_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:5]")

t_self_reasoning_dataset = load_dataset("freecs/ArtificialThinkerSet")
e_self_reasoning_dataset = load_dataset("freecs/ArtificialThinkerSet")

model, tokenizer = FastMistralModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


user_key = "query"
assistant_key = "response"
prompt_format = """###USER: {}
###ASSISTANT: {}"""


def formatting_prompts_func(examples):
    inputs = examples[user_key]
    outputs = examples[assistant_key]
    texts = []
    for input, output in zip(inputs, outputs):
        text = prompt_format.format(input, output)
        texts.append(text)
    return {"text": texts, }


def formatting_prompt_for_ultra_chat(examples):
    convs = examples["messages"]
    output = []
    for conv in convs:
        context = ""
        for message in conv:
            if message["role"] == "user":
                context += "###USER: " + message["content"] + "\n"
            elif message["role"] == "assistant":
                context += "###ASSISTANT: " + message["content"] + "\n"
                output.append(context)
    return Dataset.from_dict({"text": output, })

reasoning_prompt_format = """###USER: {}
###ASSISTANT: {} {}"""

def format_prompt_for_self_reasoning(examples):
    prompt = examples["prompt"]
    response = examples["response"]
    reasoning = examples["reasoning"]
    texts = []
    for p, r, re in zip(prompt, response, reasoning):
        text = reasoning_prompt_format.format(p, re, r)
        texts.append(text)
    return {"text": texts, }


t_metamath_formated_datasets = t_metamath_dataset.map(formatting_prompts_func, batched=True, remove_columns=["query", "response"])
e_metamath_formated_datasets = e_metamath_dataset.map(formatting_prompts_func, batched=True, remove_columns=["query", "response"])

user_key = "problem"
assistant_key = "solution"

t_code_formated_datasets = t_code_dataset.map(formatting_prompts_func, batched=True, remove_columns=["problem", "solution"])
e_code_formated_datasets = e_code_dataset.map(formatting_prompts_func, batched=True, remove_columns=["problem", "solution"])

user_key = "instruction"
assistant_key = "response"

# t_code2_formated_datasets = t_code2_dataset.map(formatting_prompts_func, batched=True, remove_columns=["instruction", "response"])
# e_code2_formated_datasets = e_code2_dataset.map(formatting_prompts_func, batched=True, remove_columns=["instruction", "response"])

t_ultra_chat_formated_datasets = formatting_prompt_for_ultra_chat(t_ultra_chat_dataset)
e_ultra_chat_formated_datasets = formatting_prompt_for_ultra_chat(e_ultra_chat_dataset)

t_self_reasoning_formated_datasets = t_self_reasoning_dataset.map(format_prompt_for_self_reasoning, batched=True, remove_columns=["prompt", "response", "reasoning"])
e_self_reasoning_formated_datasets = e_self_reasoning_dataset.map(format_prompt_for_self_reasoning, batched=True, remove_columns=["prompt", "response", "reasoning"])

t_list = [t_metamath_formated_datasets]
e_list = []


def add_ds(t_ds, e_ds):
    t_list.append(t_ds)
    e_list.append(e_ds)


# add_ds(t_code_formated_datasets, e_code_formated_datasets)
# add_ds(t_code2_formated_datasets, e_code2_formated_datasets)
add_ds(t_ultra_chat_formated_datasets, e_ultra_chat_formated_datasets)
# for i in range(6):
#     add_ds(t_self_reasoning_formated_datasets["train"], e_self_reasoning_formated_datasets["train"])

formated_datasets = concatenate_datasets(t_list).shuffle(seed=12)
e_formated_datasets = concatenate_datasets(e_list).shuffle(seed=12)

intermediate_size = 128
num_fff_layers = 6
activation_func = "gelu_new"
module = "up_proj"


    # target_modules=[module,],
    # intermediate_size=intermediate_size,
    # num_fff=num_fff_layers,
    # activation_func=activation_func,
    # use_gradient_checkpointing=True,    # When set to true VRAM grow during training (why?)

model = FastMistralModel.get_peft_model(
    model,
    target_modules=[
        f"layers.13.mlp.{module}",
        f"layers.14.mlp.{module}",
        f"layers.15.mlp.{module}",
        f"layers.16.mlp.{module}",
        f"layers.17.mlp.{module}",
        f"layers.18.mlp.{module}",
        f"layers.19.mlp.{module}",
    ],
    intermediate_size=intermediate_size,
    num_fff=num_fff_layers,
    activation_func=activation_func,
    use_gradient_checkpointing=True,
    random_state=12,
    max_seq_length=max_seq_length,
)
model.print_trainable_parameters()

trained_model_name = f"hug-ultra-lora-{intermediate_size}-{num_fff_layers}-{activation_func}-{module}"
# trained_model_name = f"lora-36-18-all"

batch_size = 3
steps = len(formated_datasets["text"])


print(f"Training {steps} steps the model: {trained_model_name}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    train_dataset=formated_datasets,
    eval_dataset=e_formated_datasets,
    args=TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        load_best_model_at_end=False,
        resume_from_checkpoint="outputs-"+trained_model_name,
        warmup_steps=1,
        num_train_epochs=1,
        report_to=["none"],
        max_steps=1,
        evaluation_strategy="steps",
        eval_steps=steps//batch_size,
        learning_rate=5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        bf16_full_eval=torch.cuda.is_bf16_supported(),
        fp16_full_eval=not torch.cuda.is_bf16_supported(),
        logging_steps=50//batch_size,
        optim="adamw_8bit",
        # max_steps=steps//batch_size,
        save_total_limit=1,
        save_strategy="steps",
        save_steps=steps//batch_size,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=12,
        output_dir="./train_mistral"+"/outputs-"+trained_model_name,
    ),
)
trainer.train(resume_from_checkpoint=True)
e = trainer.evaluate()
print(e)
# model.save_pretrained("fff-mistral-"+trained_model_name)
