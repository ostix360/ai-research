from transformers import BartConfig, MixtralConfig


class MixBartConfig(BartConfig):
    def __init__(
            self,
            num_expert=4,
            dead_zone=0.1,
            sliding_window=1024,
            mixtral_config: MixtralConfig = None,
            **kwargs):
        super().__init__(**kwargs)
        self.num_expert = num_expert
        self.dead_zone = dead_zone
        self.sliding_window = sliding_window
        self.train_norm = True
        # check if mixtral_config has the same attributes as BartConfig
        if mixtral_config is not None:
            if isinstance(mixtral_config, dict):
                mixtral_config = MixtralConfig(**mixtral_config)
            assert mixtral_config.vocab_size == self.vocab_size
            assert mixtral_config.hidden_size == self.d_model
            assert mixtral_config.pad_token_id == self.pad_token_id
            assert mixtral_config.bos_token_id == self.bos_token_id
            assert mixtral_config.eos_token_id == self.eos_token_id
            assert mixtral_config.sliding_window == self.sliding_window
            assert mixtral_config.num_local_experts == self.num_expert
        else:
            mixtral_config = MixtralConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.d_model,
                max_position_embeddings=self.max_position_embeddings,
                num_hidden_layers=self.decoder_layers,
                num_key_value_heads=4,
                intermediate_size=self.decoder_ffn_dim * 2,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                sliding_window=self.sliding_window,
                num_local_experts=self.num_expert,
            )
        self.mixtral_config = mixtral_config
        self.auto_map = {
            "AutoConfig": "configuration_mixbart.MixBartConfig",
            "AutoModelForSeq2SeqLM": "modeling_mixbart.MixBart"
        }

    @classmethod
    def from_bart_config(cls, config: BartConfig, **kwargs):
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            encoder_ffn_dim=config.encoder_ffn_dim,
            decoder_ffn_dim=config.decoder_ffn_dim,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            encoder_attention_heads=config.encoder_attention_heads,
            decoder_attention_heads=config.decoder_attention_heads,
            encoder_layerdrop=config.encoder_layerdrop,
            decoder_layerdrop=config.decoder_layerdrop,
            attention_dropout=config.attention_dropout,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
            max_position_embeddings=config.max_position_embeddings,
            init_std=config.init_std,
            classifier_dropout=config.classifier_dropout,
            num_labels=config.num_labels,
            is_encoder_decoder=config.is_encoder_decoder,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            scale_embedding=config.scale_embedding,
            use_cache=config.use_cache,
            **kwargs,
        )
