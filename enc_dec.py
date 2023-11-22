from torch import nn
import torch
from transformers import AutoModel, AutoModelForCausalLM, GPT2LMHeadModel
from transformers.models.bert.modeling_bert import BertModel
import datasets
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling


class EncDec(nn.Module):
    def __init__(self, enc_model: str, dec_model: str) -> None:
        super().__init__()
        self.encoder: BertModel = AutoModel.from_pretrained(enc_model)
        self.decoder: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(dec_model, add_cross_attention=True)
        self.adapter = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

    def forward(self, input_ids, attention_mask, labels=None, enc_input_ids=None):
        # Pass input through encoder
        encoder_outputs = self.encoder(input_ids=enc_input_ids, attention_mask=attention_mask)

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
                                       labels=labels)
        return decoder_outputs

    def _get_name(self):
        return f"{self.decoder._get_name()}"


enc_model = "bert-base-uncased"
dec_model = "gpt2"
model = EncDec(enc_model, dec_model)

t_dataset = datasets.load_dataset("wikipedia", "20220301.simple", split="train[:5000]")
e_dataset = datasets.load_dataset("wikipedia", "20220301.simple", split="train[-50:]")


cutoff_len = 512


enc_tokenizer = AutoTokenizer.from_pretrained(enc_model)
dec_tokenizer = AutoTokenizer.from_pretrained(dec_model)
dec_tokenizer.pad_token = dec_tokenizer.eos_token

def tokenize(prompt):
    enc_input_ids = enc_tokenizer(prompt, truncation=True, max_length=cutoff_len)["input_ids"]
    enc_input_ids = torch.tensor(enc_input_ids)
    dec_input_ids = dec_tokenizer(prompt, truncation=True, max_length=cutoff_len)["input_ids"]
    # dec_input_ids = [dec_tokenizer.pad_token] * (cutoff_len - len(dec_input_ids)) + dec_input_ids
    labels = [1] * len(dec_input_ids)
    dec_input_ids = torch.tensor(dec_input_ids)
    return {
        "input_ids": dec_input_ids,
        "labels": labels,
        "attention_mask": enc_input_ids.ne(enc_tokenizer.pad_token_id),
        "enc_input_ids": enc_input_ids,
    }


def tokenize_func(data):
    return tokenize(data["text"])


tokenized_datasets = t_dataset.map(tokenize_func, remove_columns=["text", "title", "id", "url"])
e_tokenized_datasets = e_dataset.map(tokenize_func, remove_columns=["text", "title", "id", "url"])

print(f"The column names are: {list(tokenized_datasets.features.keys())}")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=25,
    num_train_epochs=1,
    save_steps=5000,
    eval_steps=50,
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
    tokenizer=enc_tokenizer,
    data_collator=DataCollatorForLanguageModeling(dec_tokenizer, mlm=False),
)


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze attention layers
    for n, p in model.named_parameters():
        if "crossattention" in n or "c_proj" in n or "c-attn" in n:
            p.requires_grad = True


freeze_params(model.encoder)
# freeze_params(model.decoder)
nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {nb_trainable_params}")
trainer.train()
