import os
from typing import Union

from transformers import PretrainedConfig


class EncDecConfig(PretrainedConfig):
    model_type = "enc_dec"
    is_composition = True

    def __init__(self, enc_model: str, dec_model: str, **kwargs):
        super().__init__(**kwargs)
        self.enc_model = enc_model
        self.dec_model = dec_model

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        super().save_pretrained(save_directory, push_to_hub, **kwargs)