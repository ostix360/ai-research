import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastMistralModel


model_name = "unsloth/mistral-7b-bnb-4bit"
max_seq_length = 4096
dtype = None
load_in_4bit = True

t_code_dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[:70000]")
e_code_dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[-100:]")

t_metamath_dataset = load_dataset("meta-math/MetaMathQA", split="train[:100000]")
e_metamath_dataset = load_dataset("meta-math/MetaMathQA", split="train[-100:]")

t_code2_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train[:100000]")
e_code2_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train[-100:]")

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


t_metamath_formated_datasets = t_metamath_dataset.map(formatting_prompts_func, batched=True, remove_columns=["query", "response"])
e_metamath_formated_datasets = e_metamath_dataset.map(formatting_prompts_func, batched=True, remove_columns=["query", "response"])

user_key = "problem"
assistant_key = "solution"

t_code_formated_datasets = t_code_dataset.map(formatting_prompts_func, batched=True, remove_columns=["problem", "solution"])
e_code_formated_datasets = e_code_dataset.map(formatting_prompts_func, batched=True, remove_columns=["problem", "solution"])

user_key = "instruction"
assistant_key = "response"

t_code2_formated_datasets = t_code2_dataset.map(formatting_prompts_func, batched=True, remove_columns=["instruction", "response"])
e_code2_formated_datasets = e_code2_dataset.map(formatting_prompts_func, batched=True, remove_columns=["instruction", "response"])

formated_datasets = concatenate_datasets([t_metamath_formated_datasets, t_code_formated_datasets, t_code2_formated_datasets])
e_formated_datasets = concatenate_datasets([e_metamath_formated_datasets, e_code_formated_datasets, e_code2_formated_datasets])

model = FastMistralModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", # attention
                    "gate_proj", "up_proj", "down_proj", ], # FFN
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
    random_state=12,
    max_seq_length=max_seq_length,
)
model.print_trainable_parameters()

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    labels = np.where(labels == -100, 0, labels)
    accuracy = np.array([])
    for i in range(len(predictions)):
        valid_indices = np.where((predictions[i] != 0) & (labels[i] != 0))
        valid_predictions = predictions[i][valid_indices]
        valid_labels = labels[i][valid_indices]
        correct_predictions = np.sum(valid_predictions == valid_labels)
        total_predictions = len(valid_predictions)
        accuracy = np.append(accuracy, correct_predictions / total_predictions)
    print("Accuracy: ", accuracy.mean())
    return {"accuracy": accuracy.mean()}

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset_text_field="text",
    train_dataset=formated_datasets,
    eval_dataset=e_formated_datasets,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=5,
        num_train_epochs=1,
        report_to=["none"],
        evaluation_strategy="steps",
        eval_steps=10000,
        learning_rate=5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        bf16_full_eval= torch.cuda.is_bf16_supported(),
        fp16_full_eval=not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        optim="adamw_8bit",
        save_total_limit=2,
        save_strategy="steps",
        save_steps=10000,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=12,
        output_dir="outputs",
    ),
)

trainer.evaluate()
trainer.train()
trainer.evaluate()
trainer.save_model("my_mistral-7b-bnb-4bit")
