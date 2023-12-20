import os
from typing import Union, Optional, Callable

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, AutoConfig, PreTrainedModel, AutoModel
from transformers.modeling_outputs import Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.bert.modeling_bert import BertModel
from transformers.utils import is_torch_fx_proxy

from .configuration_enc_dec import EncDecConfig


class EncDec(PreTrainedModel):
    config_class = EncDecConfig

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        if config._name_or_path == "":
            self.enc_model = config.enc_model
            self.dec_model = config.dec_model
            bert_config = AutoConfig.from_pretrained(self.enc_model, torch_dtype=torch.bfloat16)
            bert_config.use_flash_attn = True

            self.encoder = BertModel.from_pretrained(self.enc_model, config=bert_config, )
            self.decoder: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(self.dec_model, add_cross_attention=True)
        else:
            # TODO, useless loading and warning
            self.encoder = AutoModel.from_pretrained(config.enc_model)
            self.decoder = AutoModelForCausalLM.from_pretrained(config.dec_model, add_cross_attention=True)
        self.adapter = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size, bias=False)
        self.lm_head = nn.Linear(self.config.decoder_n_embd, self.config.decoder_vocab_size, bias=False)
        print("post_init")
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids,
            attention_mask,
            dec_attention_mask,
            labels=None,
            dec_input_ids=None,
            use_cache=None,
            past_key_values=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict: Optional[bool] = None,
            encoder_outputs=None,
    ):
        if dec_input_ids is None and input_ids is None:
            raise ValueError("Input_ids and decoder_input_ids cannot be both None")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        # Pass input through encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        # Adapter brings the encoder outputs to the correct dimension for the decoder
        encoder_hidden_states = self.adapter(encoder_outputs.last_hidden_state)

        # Pass adapter outputs and decoder_input_ids to the decoder
        # In this case, "encoder_hidden_states" will be used as cross-attention "encoder_attention_mask"
        # You have to manage them according to your use-case
        # check len label and input_ids
        if labels is not None:
            if len(labels) != len(dec_input_ids):
                print(len(labels), len(dec_input_ids))
                raise ValueError("Input_ids and labels should have the same length")

        decoder_outputs = self.decoder(
            input_ids=dec_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            output_hidden_states=True,
            output_attentions=True,
            attention_mask=dec_attention_mask
        )
        lm_logits = self.lm_head(decoder_outputs.hidden_states[-1])

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def can_generate(cls) -> bool:
        return True

    def _get_name(self):
        return f"{self.encoder._get_name()}To{self.decoder._get_name()}"

    # copied from bart
    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        decoder_input_ids = kwargs["dec_input_ids"]
        first_input_ids = input_ids[0][0].view(1, -1)
        input_ids = torch.cat((first_input_ids, decoder_input_ids, input_ids[0][1:].view(1, -1)), dim=1)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        # if encoder_outputs is not None:
        #     input_ids = None
        # first step, decoder_cached_states are empty
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "dec_input_ids": input_ids,
            "attention_mask": attention_mask,
            "dec_attention_mask": decoder_attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    # copied from mt5
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    # copied from mt5
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.decoder_pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In MT5 it is usually set to the pad_token_id. "
                "See MT5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


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
        self.encoder.config.save_pretrained(save_directory)
        self.decoder.config.save_pretrained(save_directory)
        super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)
