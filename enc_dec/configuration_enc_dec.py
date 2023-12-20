import os
from typing import Union

from transformers import PretrainedConfig, AutoConfig


class EncDecConfig(PretrainedConfig):
    model_type = "enc_dec"
    is_composition = True

    def __init__(self, enc_model: str, dec_model: str, **kwargs):
        super().__init__(**kwargs)
        decoder_config = AutoConfig.from_pretrained(dec_model)
        self.enc_model = enc_model
        self.dec_model = dec_model
        self.is_encoder_decoder = True
        self.decoder_pad_token_id = decoder_config.eos_token_id
        self.decoder_n_embd = decoder_config.n_embd
        self.decoder_vocab_size = decoder_config.vocab_size
        self.decoder_start_token_id = decoder_config.bos_token_id
        self.tie_word_embeddings = False
        self.use_cache = True
        self.auto_map = {
            "AutoModelForCausalLM": "modeling_enc_dec.EncDec",
            "AutoConfig": "configuration_enc_dec.EncDecConfig",
        }

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        super().save_pretrained(save_directory, push_to_hub, **kwargs)
