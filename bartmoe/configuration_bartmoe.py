from transformers import BartConfig


class BartMoeConfig(BartConfig):
    def __init__(
            self,
            num_expert=4,
            dead_zone=0.1,
            sliding_window=1024,
            **kwargs):
        super().__init__(**kwargs)
        self.num_expert = num_expert
        self.dead_zone = dead_zone
        self.sliding_window = sliding_window

    @classmethod
    def from_bart_config(cls, config: BartConfig):
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
        )
