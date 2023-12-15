import os
from typing import Union, Optional, Callable

import torch
from torch import nn
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, AutoConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertModel


class EncDec(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.enc_model = config.enc_model
        self.dec_model = config.dec_model
        bert_config = AutoConfig.from_pretrained(self.enc_model, torch_dtype=torch.bfloat16)
        bert_config.use_flash_attn = True

        self.encoder = BertModel.from_pretrained(self.enc_model, config=bert_config, )
        self.decoder: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(self.dec_model, add_cross_attention=True)
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
        self.decoder.save_pretrained(save_directory+"/"+self.dec_model, is_main_process, state_dict, save_function, push_to_hub,
                                     max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)
        self.encoder.save_pretrained(save_directory+"/"+self.enc_model, is_main_process, state_dict, save_function, push_to_hub,
                                     max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)
        super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)
