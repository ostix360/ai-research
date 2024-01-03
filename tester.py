import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

# from enc_dec.modeling_enc_dec import EncDec
from bartmoe.bart_patch import apply_sliding_window_patch
from mixbart.modeling_mixbart import MixBart

apply_sliding_window_patch()

checkpoint = "./mixbart_moe_experts"

model = MixBart.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prompt = "###USER: Solve the following equation: 2x + 3 = 5"

input_ids = tokenizer(prompt, return_tensors="pt")

enc_input_ids = input_ids["input_ids"].to("cuda")
attention_mask = input_ids["attention_mask"].to("cuda")
model.to("cuda")

model.train()

inputs = {
    "attention_mask": attention_mask,
    "input_ids": enc_input_ids,
    "expert_train": [2]
}
out = model.generate(**inputs, max_length=400, repetition_penalty=1.1, do_sample=True, top_k=50, top_p=0.6, temperature=0.5, num_return_sequences=1,)
text = ""
for i in out:
    text += tokenizer.decode(i)
print(text)

print("Done")