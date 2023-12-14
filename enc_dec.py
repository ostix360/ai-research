# import fused_dense_lib
import os
from typing import Union, Optional, Callable

from torch import nn
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, GPT2LMHeadModel, AutoConfig, PreTrainedModel, PretrainedConfig
from transformers.models.bert import modeling_bert
from transformers.models.gpt2 import modeling_gpt2
from transformers.models.bert.modeling_bert import BertModel
import datasets
# import bert_fa2
# from flash_attn.models.bert import BertModel
from flash_attn.models.gpt import GPTModel, GPTLMHeadModel
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
# from flash_attn.ops.fused_dense import ColumnParallelLinear

import utils


# modeling_bert.BertSelfAttention = bert_fa2.BertFlashAttention2Attention
# import gpt_fa2
# gpt_fa2.apply_patch()

class EncDecConfig(PretrainedConfig):
    model_type = "enc_dec"
    is_composition = True

    def __init__(self, enc_model: str, dec_model: str, **kwargs):
        super().__init__(**kwargs)
        self.enc_model = enc_model
        self.dec_model = dec_model

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        self.encoder.save_pretrained(save_directory, push_to_hub, **kwargs)
        self.decoder.save_pretrained(save_directory, push_to_hub, **kwargs)


class EncDec(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        enc_model = config.enc_model
        dec_model = config.dec_model
        bert_config = AutoConfig.from_pretrained(enc_model, torch_dtype=torch.bfloat16)
        bert_config.use_flash_attn = True

        self.encoder = BertModel.from_pretrained(enc_model, config=bert_config, )
        self.decoder: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(dec_model, add_cross_attention=True)
        self.adapter = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

    def forward(self, input_ids, enc_attention_mask, attention_mask, labels=None, enc_input_ids=None):
        # Pass input through encoder
        encoder_outputs = self.encoder(input_ids=enc_input_ids, attention_mask=enc_attention_mask)

        # Adapter brings the encoder outputs to the correct dimension for the decoder
        encoder_hidden_states = self.adapter(encoder_outputs.last_hidden_state)

        # Pass adapter outputs and decoder_input_ids to the decoder
        # In this case, "encoder_hidden_states" will be used as cross-attention "encoder_attention_mask"
        # You have to manage them according to your use-case
        # check len label and input_ids
        if labels is not None:
            if len(labels) != len(input_ids):
                print(len(labels), len(input_ids))
                raise ValueError("Input_ids and labels should have the same length")

        decoder_outputs = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                                       labels=labels, attention_mask=attention_mask)
        return decoder_outputs

    def _get_name(self):
        return f"{self.decoder._get_name()}"

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            max_shard_size: Union[int, str] = "5GB",
            safe_serialization: bool = True,
            variant: Optional[str] = None,
            token: Optional[Union[str, bool]] = None,
            save_peft_format: bool = True,
            **kwargs,
    ):
        self.decoder.save_pretrained(dec_model+"/"+save_directory, is_main_process, state_dict, save_function, push_to_hub,
                                     max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)
        self.encoder.save_pretrained(enc_model+"/"+save_directory, is_main_process, state_dict, save_function, push_to_hub,
                                     max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)


enc_model = "bert-base-uncased"
dec_model = "gpt2"
encdec_config = EncDecConfig(enc_model, dec_model)
model = EncDec(encdec_config)

t_dataset = datasets.load_dataset("wikipedia", "20220301.simple", split="train[:30000]")
e_dataset = datasets.load_dataset("wikipedia", "20220301.simple", split="train[-50:]")

cutoff_len = 512
bert_config = AutoConfig.from_pretrained(dec_model, torch_dtype=torch.bfloat16)
bert_config.use_flash_attn = True

# model = GPTLMHeadModel.from_pretrained(dec_model, config=bert_config, strict=False)

enc_tokenizer = AutoTokenizer.from_pretrained(enc_model)
dec_tokenizer = AutoTokenizer.from_pretrained(dec_model)
dec_tokenizer.pad_token = dec_tokenizer.eos_token


def tokenize(prompt):
    enc_input_ids = enc_tokenizer(prompt, truncation=True, max_length=cutoff_len)["input_ids"]
    enc_input_ids = enc_input_ids + [enc_tokenizer.pad_token_id] * (cutoff_len - len(enc_input_ids))
    enc_input_ids = torch.tensor(enc_input_ids)
    dec_input_ids = dec_tokenizer(prompt, padding="max_length", truncation=True, max_length=cutoff_len)["input_ids"]

    labels = [1] * len(dec_input_ids)
    dec_input_ids = torch.tensor(dec_input_ids)
    return {
        "input_ids": dec_input_ids,
        "attention_mask": dec_input_ids.ne(dec_tokenizer.pad_token_id),
        "labels": labels,
        "enc_attention_mask": enc_input_ids.ne(enc_tokenizer.pad_token_id),
        "enc_input_ids": enc_input_ids,
    }


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
    return tokenize(data["text"])


tokenized_datasets = t_dataset.map(tokenize_func, remove_columns=["text", "title", "id", "url"])
e_tokenized_datasets = e_dataset.map(tokenize_func, remove_columns=["text", "title", "id", "url"])

print(f"The column names are: {list(tokenized_datasets.features.keys())}")
data_collator = DataCollatorForLanguageModeling(dec_tokenizer, mlm=False)
train_dataloader = DataLoader(
    tokenized_datasets, batch_size=1, shuffle=True, collate_fn=data_collator
)

# Debug
# batch = utils.debug_data_processing(train_dataloader)
# bert_config = AutoConfig.from_pretrained(dec_model, torch_dtype=torch.bfloat16, use_cache=False)
# bert_config.use_flash_attn = True
#
# bert_model = AutoModelForCausalLM.from_pretrained(dec_model, config=bert_config)
# bert_model = bert_model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
# input_ids = batch["input_ids"].to(device=torch.device("cuda"))
#
#
# out = bert_model(input_ids, )

training_args = TrainingArguments(
    output_dir="result",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=10,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=25,
    num_train_epochs=1,
    eval_steps=50,
    bf16=True,
    bf16_full_eval=True,
    # fp16_backend="apex",
    # fp16=True,
    # fp16_full_eval=True,
    # torch_compile=True,
    # half_precision_backend="apex",
    save_strategy="steps",
    save_steps=3000,
    warmup_steps=2,
    learning_rate=2e-5,
    save_total_limit=1,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=e_tokenized_datasets,
    tokenizer=dec_tokenizer,
    data_collator=data_collator,
)


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze attention layers
    # for n, p in model.named_parameters():
    #     if "crossattention" in n or "c_proj" in n or "c-attn" in n:
    #         p.requires_grad = True


# freeze_params(model.encoder)
# freeze_params(model.decoder)
nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {nb_trainable_params}")
# print the model architecture (state_dict)

trainer.train()
