import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

# from enc_dec.modeling_enc_dec import EncDec
from bartmoe.bart_patch import apply_sliding_window_patch

apply_sliding_window_patch()

checkpoint = "./final_result_code"

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prompt = "###USER: You are tasked with implementing a function that takes a list of integers and returns the average of the list."

input_ids = tokenizer(prompt, return_tensors="pt")

enc_input_ids = input_ids["input_ids"].to("cuda")
attention_mask = input_ids["attention_mask"].to("cuda")
model.to("cuda")

model.train()

inputs = {
    "attention_mask": attention_mask,
    "input_ids": enc_input_ids,
}
out = model.generate(**inputs, max_length=800, repetition_penalty=1.1, do_sample=True, top_k=50, top_p=0.6, temperature=0.5, num_return_sequences=1,)
text = ""
for i in out:
    text += tokenizer.decode(i)
print(text)

print("Done")