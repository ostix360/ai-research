import torch
from datasets import load_dataset, concatenate_datasets
from peft import TaskType

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling, \
    BitsAndBytesConfig
from trl import SFTTrainer
from transformers import TrainingArguments

import bitsandbytes as bnb

import fff_mistral_patch
from train_mistral.fff.config import fffConfig
from train_mistral.fff.model import patch_peft_for_loading

from peft.mapping import get_peft_model
from peft import prepare_model_for_kbit_training

# fff_mistral_patch.patch_to_fff_mistral()
fff_mistral_patch.patch_to_unsloth_mistral()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model_name = "unsloth/mistral-7b-bnb-4bit"
max_seq_length = 4096
dtype = torch.bfloat16
load_in_4bit = False

t_code_dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[:7000]")
e_code_dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[-100:]")

t_metamath_dataset = load_dataset("meta-math/MetaMathQA", split="train[:10000]")
e_metamath_dataset = load_dataset("meta-math/MetaMathQA", split="train[-100:]")

t_code2_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train[:10000]")
e_code2_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train[-100:]")

config = AutoConfig.from_pretrained(model_name)
# config.fff_num_layers = 1
# config.fff_intermediate_size = [512]
# config.fff_hidden_act = ["silu"]
# config.fff_bias = [False]

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map=0,
)

fff_config = fffConfig(
    target_modules=["up_proj"],
    intermediate_size=128,
    num_fff=1,
    activation_func="silu",
    task_type=TaskType.CAUSAL_LM,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

model = get_peft_model(model, fff_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

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


t_metamath_formated_datasets = t_metamath_dataset.map(formatting_prompts_func, batched=True,
                                                      remove_columns=["query", "response"])
e_metamath_formated_datasets = e_metamath_dataset.map(formatting_prompts_func, batched=True,
                                                      remove_columns=["query", "response"])

user_key = "problem"
assistant_key = "solution"

t_code_formated_datasets = t_code_dataset.map(formatting_prompts_func, batched=True,
                                              remove_columns=["problem", "solution"])
e_code_formated_datasets = e_code_dataset.map(formatting_prompts_func, batched=True,
                                              remove_columns=["problem", "solution"])

user_key = "instruction"
assistant_key = "response"

t_code2_formated_datasets = t_code2_dataset.map(formatting_prompts_func, batched=True,
                                                remove_columns=["instruction", "response"])
e_code2_formated_datasets = e_code2_dataset.map(formatting_prompts_func, batched=True,
                                                remove_columns=["instruction", "response"])

formated_datasets = concatenate_datasets(
    [t_metamath_formated_datasets, t_code_formated_datasets, t_code2_formated_datasets])
e_formated_datasets = concatenate_datasets(
    [e_metamath_formated_datasets, e_code_formated_datasets, e_code2_formated_datasets])

#
# def freeze_params(model):
#     for name, param in model.named_parameters():
#         pass
#         if "fff" in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False


# freeze_params(model)


# print the trainable parameters
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./fff_mistral",
    overwrite_output_dir=True,
    num_train_epochs=1,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    bf16=True,
    bf16_full_eval=True,
    optim="adamw_8bit",
    save_steps=10000,
    max_steps=500,
    eval_steps=1000,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to=["none"],
)

trainer = SFTTrainer(
    tokenizer=tokenizer,
    model=model,
    max_seq_length=max_seq_length,
    dataset_text_field="text",
    args=training_args,
    train_dataset=formated_datasets,
    eval_dataset=e_formated_datasets,
)

trainer.train()
trainer.save_model("./fff_mistral")
