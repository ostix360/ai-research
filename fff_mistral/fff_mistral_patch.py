import types
import warnings
from typing import Optional, Tuple

import torch
from peft import TaskType
from torch import nn
from transformers import MistralConfig, set_seed
from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import MISTRAL_ATTENTION_CLASSES, MistralRMSNorm

from train_mistral.fff.config import fffConfig


class MistralMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = MistralMLP(config.hidden_size, config.intermediate_size, config.hidden_act,
                              bias=False)  # Default mistral ffn
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.fff_num_layers > 0:
            assert len(config.fff_intermediate_size) == config.fff_num_layers
            assert len(config.fff_hidden_act) == config.fff_num_layers
            assert len(config.fff_bias) == config.fff_num_layers
            self.fff = nn.ModuleList(
                [
                    MistralMLP(
                        config.hidden_size,
                        config.fff_intermediate_size[i],
                        config.fff_hidden_act[i],
                        config.fff_bias[i],
                    )
                    for i in range(config.fff_num_layers)
                ]
            )
            self.fff_layernorms = nn.ModuleList(
                [MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(config.fff_num_layers)]
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        for fff, fff_layernorm in zip(self.fff, self.fff_layernorms):
            residual = hidden_states
            hidden_states = fff_layernorm(hidden_states)
            hidden_states = fff(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def _init_weights(self, module):
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, MistralRMSNorm):
        module.weight.data.fill_(1.0)
        if module.bias is not None:
            module.bias.data.zero_()


def patch_to_fff_mistral():
    from transformers.models.mistral import modeling_mistral
    modeling_mistral.MistralMLP = MistralMLP
    modeling_mistral.MistralDecoderLayer = MistralDecoderLayer
    modeling_mistral.PreTrainedModel._init_weights = _init_weights


@staticmethod
def get_peft_model(
        model,
        intermediate_size=256,
        num_fff=1,
        target_modules=["gate_proj", "up_proj"],
        activation_func="silu",
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=2048,  # not used anymore
        **kwargs,
):
    from unsloth.models._utils import __version__
    from unsloth.kernels import apply_lora_mlp, apply_lora_qkv, apply_lora_o
    from unsloth.models._utils import prepare_model_for_kbit_training
    from transformers.models.llama.modeling_llama import logger
    from peft import get_peft_model as _get_peft_model

    assert (max_seq_length <= model.max_seq_length)

    set_seed(random_state)

    accepted_modules = frozenset(("gate_proj", "up_proj",), )
    model.config.update({"unsloth_version": __version__})
    for module in target_modules:
        assert (module in accepted_modules)
    pass

    # Get fff
    fff_config = fffConfig(
        intermediate_size=intermediate_size,
        num_fff=num_fff,
        target_modules=target_modules,
        activation_func=activation_func,
        task_type=TaskType.CAUSAL_LM,
        **kwargs,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_reentrant=True,
    )
    model = _get_peft_model(model, fff_config)

    # Do patching
    n_mlp = 0
    n_qkv = 0
    n_o = 0
    for idx, layer in enumerate(model.model.model.layers):

        # MLP patching
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        down_proj = layer.mlp.down_proj

        if hasattr(gate_proj, "lora_A") and \
                hasattr(up_proj, "lora_A") and \
                hasattr(down_proj, "lora_A") and \
                (gate_proj.base_layer if hasattr(gate_proj, "base_layer") else gate_proj).bias is None and \
                (up_proj.base_layer if hasattr(up_proj, "base_layer") else up_proj).bias is None and \
                (down_proj.base_layer if hasattr(down_proj, "base_layer") else down_proj).bias is None:

            # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            layer.mlp.forward = types.MethodType(apply_lora_mlp, layer.mlp)
            n_mlp += 1
        else:
            logger.warning_once(
                "Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters\n" \
                "are not enabled or a bias term (like in Qwen) is used."
            )
        pass

        # QKV attention patching
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        if hasattr(q_proj, "lora_A") and \
                hasattr(k_proj, "lora_A") and \
                hasattr(v_proj, "lora_A") and \
                (q_proj.base_layer if hasattr(q_proj, "base_layer") else q_proj).bias is None and \
                (k_proj.base_layer if hasattr(k_proj, "base_layer") else k_proj).bias is None and \
                (v_proj.base_layer if hasattr(v_proj, "base_layer") else v_proj).bias is None:

            layer.self_attn.apply_qkv = apply_lora_qkv
            n_qkv += 1
        else:
            logger.warning_once(
                "Unsloth cannot patch Attention layers with our manual autograd engine since either LoRA adapters\n" \
                "are not enabled or a bias term (like in Qwen) is used."
            )
        pass

        # O attention patching
        o_proj = layer.self_attn.o_proj
        if hasattr(o_proj, "lora_A") and \
                (o_proj.base_layer if hasattr(o_proj, "base_layer") else o_proj).bias is None:

            layer.self_attn.apply_o = apply_lora_o
            n_o += 1
        else:
            logger.warning_once(
                "Unsloth cannot patch O projection layer with our manual autograd engine since either LoRA adapters\n" \
                "are not enabled or a bias term (like in Qwen) is used."
            )
        pass
    pass

    logger.warning_once(
        f"Unsloth {__version__} patched {len(model.model.model.layers)} layers with " \
        f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers.",
    )

    # Patch cross entropy loss labels
    # Fixes https://github.com/unslothai/unsloth/issues/10
    max_seq_length = model.max_seq_length
    extra_ignored_labels = torch.full((max_seq_length, 1), -100, device="cuda")
    model.model.extra_ignored_labels = extra_ignored_labels
    internal_model = model
    while hasattr(internal_model, "model"):
        internal_model.max_seq_length = max_seq_length
        internal_model = internal_model.model
    pass
    internal_model.max_seq_length = max_seq_length
    return model


def patch_to_unsloth_mistral():
    from unsloth.models import llama
    from train_mistral.fff.model import patch_peft_for_loading
    patch_peft_for_loading()
    llama.FastLlamaModel.get_peft_model = get_peft_model
