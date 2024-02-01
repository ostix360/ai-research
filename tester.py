import torch
from datasets import load_dataset
from peft import PeftMixedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments
from trl import SFTTrainer

from train_mistral import fff_mistral_patch
from utils import TokenStoppingCriteria

# from enc_dec.modeling_enc_dec import EncDec
# from bartmoe.bart_patch import apply_sliding_window_patch

# apply_sliding_window_patch()



fff_mistral_patch.patch_to_unsloth_mistral()

checkpoint = "unsloth/mistral-7b-bnb-4bit"


model = AutoModelForCausalLM.from_pretrained(checkpoint,).eval()
model = PeftMixedModel.from_pretrained(model, "./train_mistral/fff-mistral-hug-ultra-lora-128-6-gelu_new-up_proj", "default")
model.load_adapter("./train_mistral/fff-mistral-hug-ultra-lora-128-6-gelu_new-up_proj/lora", "lora")
model.set_adapter(["lora", "default"])
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prompt = "###USER: Solve the following equation: 2x + 3 = 5 - x and write a python code that solve this equation for checking."

input_ids = tokenizer(prompt, return_tensors="pt")

enc_input_ids = input_ids["input_ids"].to("cuda")
attention_mask = input_ids["attention_mask"].to("cuda")
# model.to("cuda")
#
# model.train()

inputs = {
    "attention_mask": attention_mask,
    "input_ids": enc_input_ids,
    # "expert_train": [2]
}

stop_token_id = tokenizer.encode("###USER:")[:]
stopping_criteria = TokenStoppingCriteria(stop_token_id)

# e_ultra_chat_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:5]")
#
# def formatting_prompt_for_ultra_chat(examples):
#     convs = examples["messages"]
#     output = []
#     for conv in convs:
#         context = ""
#         for message in conv:
#             if message["role"] == "user":
#                 context += "###USER: " + message["content"] + "\n"
#             elif message["role"] == "assistant":
#                 context += "###ASSISTANT: " + message["content"] + "\n"
#                 output.append(context)
#     from datasets import Dataset
#     return Dataset.from_dict({"text": output, })
#
#
# e_ultra_chat_formated_datasets = formatting_prompt_for_ultra_chat(e_ultra_chat_dataset)
# # tokenizer.padding_side = "left"
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     dataset_text_field="text",
#     max_seq_length=2048,
#     train_dataset=e_ultra_chat_formated_datasets,
#     eval_dataset=e_ultra_chat_formated_datasets,
#     args=TrainingArguments(
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=5,
#         gradient_accumulation_steps=1,
#         load_best_model_at_end=False,
#         warmup_steps=1,
#         num_train_epochs=1,
#         report_to=["none"],
#         evaluation_strategy="steps",
#         eval_steps=10,
#         learning_rate=5e-5,
#         fp16=not torch.cuda.is_bf16_supported(),
#         bf16=torch.cuda.is_bf16_supported(),
#         bf16_full_eval=torch.cuda.is_bf16_supported(),
#         fp16_full_eval=not torch.cuda.is_bf16_supported(),
#         logging_steps=50,
#         optim="adamw_8bit",
#         max_steps=50,
#         save_total_limit=1,
#         save_strategy="steps",
#         save_steps=50,
#         weight_decay=0.01,
#         lr_scheduler_type="linear",
#         seed=12,
#         output_dir="outputs-eval",
#     ),
# )
#
# e = trainer.evaluate()
#
# print(e)

out = model.generate(**inputs,max_length=800,do_sample=True, repetition_penalty=1.1, top_k=50, top_p=0.6, temperature=0.5, num_return_sequences=1,stopping_criteria=[stopping_criteria])
text = ""
for i in out:
    text += tokenizer.decode(i)
print(text)

print("Done")