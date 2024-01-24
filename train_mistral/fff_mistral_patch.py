import types
import warnings
from typing import Optional, Tuple, Any, List, Union

import torch
from peft import TaskType, __version__
from peft.tuners.mixed.model import Configs, COMPATIBLE_TUNER_TYPES
from torch import nn
from transformers import MistralConfig, set_seed
from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import MISTRAL_ATTENTION_CLASSES, MistralRMSNorm

from fff.config import fffConfig


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
    from peft import LoraConfig
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

    lora_target = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=32,
        bias="none",
        lora_alpha=16,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_target,
        lora_dropout=0,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_reentrant=True,
    )
    model = _get_peft_model(model, lora_config, adapter_name="lora", mixed=True)
    model.add_adapter("default", fff_config)
    model.set_adapter(["lora", "default"])

    # Do patching
    n_mlp = 0
    n_qkv = 0
    n_o = 0
    for idx, layer in enumerate(model.base_model.model.model.layers):  # base_model added for mixed peft

        # MLP patching
        # gate_proj = layer.mlp.gate_proj
        # up_proj = layer.mlp.up_proj
        # down_proj = layer.mlp.down_proj
        #
        # if hasattr(gate_proj, "lora_A") and \
        #         hasattr(up_proj, "lora_A") and \
        #         hasattr(down_proj, "lora_A") and \
        #         (gate_proj.base_layer if hasattr(gate_proj, "base_layer") else gate_proj).bias is None and \
        #         (up_proj.base_layer if hasattr(up_proj, "base_layer") else up_proj).bias is None and \
        #         (down_proj.base_layer if hasattr(down_proj, "base_layer") else down_proj).bias is None:
        #
        #     # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
        #     layer.mlp.forward = types.MethodType(apply_lora_mlp, layer.mlp)
        #     n_mlp += 1
        # else:
        logger.warning_once(
                "Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters\n"
                "are not enabled or a bias term (like in Qwen) is used."
                "This is not a problem if you are using fff config."
            )


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
        f"Unsloth {__version__} patched {len(model.base_model.model.model.layers)} layers with " \
        f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers.",
    )

    # Patch cross entropy loss labels
    # Fixes https://github.com/unslothai/unsloth/issues/10
    max_seq_length = 4096
    extra_ignored_labels = torch.full((max_seq_length, 1), -100, device="cuda")
    model.base_model.model.extra_ignored_labels = extra_ignored_labels
    internal_model = model
    has_loop = False
    while hasattr(internal_model, "model") or hasattr(internal_model, "base_model"):
        internal_model.max_seq_length = max_seq_length
        if hasattr(internal_model, "model"):
            internal_model = internal_model.model
        else:
            if has_loop:
                break
            internal_model = internal_model.base_model
            has_loop = True
    pass
    internal_model.max_seq_length = max_seq_length
    return model


pass


def _create_and_replace(
        self,
        config: Configs,
        *args: Any,
        **kwargs: Any,
) -> None:
    from peft.tuners import adalora
    from peft.tuners import lora
    from peft.tuners import loha
    from peft.tuners import lokr
    from peft.tuners import oft
    import fff
    if isinstance(config, adalora.AdaLoraConfig):
        adalora.AdaLoraModel._create_and_replace(self, config, *args, **kwargs)
    elif isinstance(config, lora.LoraConfig):
        lora.LoraModel._create_and_replace(self, config, *args, **kwargs)
    elif isinstance(config, loha.LoHaConfig):
        loha.LoHaModel._create_and_replace(self, config, *args, **kwargs)
    elif isinstance(config, lokr.LoKrConfig):
        lokr.LoKrModel._create_and_replace(self, config, *args, **kwargs)
    elif isinstance(config, oft.OFTConfig):
        oft.OFTModel._create_and_replace(self, config, *args, **kwargs)
    elif isinstance(config, fff.fffConfig):
        fff.fffModel._create_and_replace(self, config, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")


@staticmethod
def _create_new_module(config, adapter_name, target, **kwargs):
    from peft.tuners import adalora
    from peft.tuners import lora
    from peft.tuners import loha
    from peft.tuners import lokr
    from peft.tuners import oft
    import fff

    if isinstance(config, adalora.AdaLoraConfig):
        new_module = adalora.AdaLoraModel._create_new_module(config, adapter_name, target, **kwargs)
    elif isinstance(config, lora.LoraConfig):
        new_module = lora.LoraModel._create_new_module(config, adapter_name, target, **kwargs)
    elif isinstance(config, loha.LoHaConfig):
        new_module = loha.LoHaModel._create_new_module(config, adapter_name, target, **kwargs)
    elif isinstance(config, lokr.LoKrConfig):
        new_module = lokr.LoKrModel._create_new_module(config, adapter_name, target, **kwargs)
    elif isinstance(config, oft.OFTConfig):
        new_module = oft.OFTModel._create_new_module(config, adapter_name, target, **kwargs)
    elif isinstance(config, fff.fffConfig):
        new_module = fff.fffModel._create_new_module(config, adapter_name, target, **kwargs)
    else:
        raise ValueError(f"Unknown config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")
    return new_module


def _set_signature_columns_if_needed(self):
    from transformers.trainer import _is_peft_model
    import inspect
    from peft import PeftMixedModel
    if self._signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        model_to_inspect = self.model

        if _is_peft_model(self.model):
            model_to_inspect = self.model.get_base_model()
        elif isinstance(model_to_inspect, PeftMixedModel):
            model_to_inspect = model_to_inspect.base_model.model

        signature = inspect.signature(model_to_inspect.forward)
        self._signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        self._signature_columns += list(set(["label", "label_ids"] + self.label_names))


def can_return_loss(model_class):
    """
    Check if a given model can return loss.

    Args:
        model_class (`type`): The class of the model.
    """
    from transformers.utils import infer_framework
    import inspect
    from peft import PeftMixedModel
    if model_class is PeftMixedModel:
        return True
    framework = infer_framework(model_class)
    if framework == "tf":
        signature = inspect.signature(model_class.call)  # TensorFlow models
    elif framework == "pt":
        signature = inspect.signature(model_class.forward)  # PyTorch models
    else:
        signature = inspect.signature(model_class.__call__)  # Flax models

    for p in signature.parameters:
        if p == "return_loss" and signature.parameters[p].default is True:
            return True

    return False


def create_or_update_model_card(self, save_directory):
    import os
    from huggingface_hub import ModelCard
    from huggingface_hub import ModelCardData

    filename = os.path.join(save_directory, "README.md")

    card = ModelCard.load(filename) if os.path.exists(filename) else ModelCard.from_template(ModelCardData())

    card.data["library_name"] = "peft"

    model_config = getattr(self, "config", None)
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()
    if model_config is not None:
        card.data["base_model"] = model_config["_name_or_path"]

    lines = card.text.splitlines()

    quantization_config = None
    if hasattr(model_config, "quantization_config"):
        quantization_config = self.config.quantization_config.to_dict()
    training_config_text = ""
    quantization_prefix = "The following `bitsandbytes` quantization config was used during training:"
    # Adds quantization information if it was used
    if quantization_config is not None:
        training_config_text += f"\n{quantization_prefix}\n"
        training_config_text += "\n".join([f"- {name}: {value}" for name, value in quantization_config.items()])
        training_config_text += "\n"

    training_procedure_heading = "## Training procedure"
    if quantization_prefix not in lines and bool(training_config_text):
        if training_procedure_heading in lines:
            lines.insert(lines.index(training_procedure_heading) + 2, training_config_text)
        else:
            lines.append(f"{training_procedure_heading}\n{training_config_text}")

    # Adds peft version
    framework_block_heading = "### Framework versions"
    if f"- PEFT {__version__}" not in lines:
        if framework_block_heading in lines:
            lines.insert(lines.index(framework_block_heading) + 2, f"- PEFT {__version__} - PATCHED")
        else:
            lines.append(f"{framework_block_heading}\n\n- PEFT {__version__} - PATCHED")

    card.text = "\n".join(lines)
    card.save(filename)


def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[List[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        **kwargs: Any,
) -> None:
    r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            safe_serialization (`bool`, *optional*):
                Whether to save the adapter files in safetensors format, defaults to `True`.
            selected_adapters (`List[str]`,  *optional*):
                A list of adapters to be saved. If `None`, will default to all adapters.
            save_embedding_layers (`Union[bool, str]`, *optional*, defaults to `"auto"`):
                If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common
                embedding layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available.
                and automatically sets the boolean flag. This only works for 🤗 transformers models.
            is_main_process (`bool`, *optional*):
                Whether the process calling this is the main process or not. Will default to `True`. Will not save the
                checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
    import os
    from safetensors.torch import save_file as safe_save_file
    from fff.model import _get_peft_model_state_dict
    import collections
    from peft.utils import id_tensor_storage
    from peft.utils import SAFETENSORS_WEIGHTS_NAME
    from peft.utils import WEIGHTS_NAME
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

    if selected_adapters is None:
        selected_adapters = list(self.peft_config.keys())
    else:
        if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
        ):
            raise ValueError(
                f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                f" {list(self.peft_config.keys())} - got {selected_adapters}."
            )

    if is_main_process:
        os.makedirs(save_directory, exist_ok=True)
        create_or_update_model_card(self, save_directory)

    for adapter_name in selected_adapters:
        peft_config = self.peft_config[adapter_name]
        # save only the trainable weights

        output_state_dict = _get_peft_model_state_dict(
            self,
            state_dict=kwargs.get("state_dict", None),
            adapter_name=adapter_name,
            save_embedding_layers=save_embedding_layers,
        )
        output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
        os.makedirs(output_dir, exist_ok=True)

        if is_main_process and safe_serialization:
            # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving

            ptrs = collections.defaultdict(list)
            for name, tensor in output_state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors.
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            for _, names in shared_ptrs.items():
                # Here we just clone the shared tensors to avoid tensor aliasing which is
                # not supported in safetensors.
                for shared_tensor_name in names[1:]:
                    output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()


            safe_save_file(
                output_state_dict,
                os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
        elif is_main_process:
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = (
                self.base_model.__dict__.get("name_or_path", None)
                if peft_config.is_prompt_learning
                else self.base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True

        if peft_config.task_type is None:
            # deal with auto mapping
            base_model_class = self._get_base_model_class(
                is_prompt_tuning=peft_config.is_prompt_learning,
            )
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }
        else:
            auto_mapping_dict = None

        if is_main_process:
            peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
        peft_config.inference_mode = inference_mode




def patch_to_unsloth_mistral():
    from unsloth.models import llama
    import fff
    from fff.model import patch_peft_for_loading
    from peft.tuners import mixed
    from peft import PeftType
    from typing import Union
    import peft
    patch_peft_for_loading()
    llama.FastLlamaModel.get_peft_model = get_peft_model

    # Mix patch
    peft.mixed_model.COMPATIBLE_TUNER_TYPES = mixed.COMPATIBLE_TUNER_TYPES + (PeftType.FFF,)
    mixed.model.PREFIXES.append("fff")
    mixed.model.Layers = mixed.model.Layers + (fff.fffLayer,)
    mixed.model.Configs = Union[mixed.model.Configs, fffConfig]
    mixed.model.MixedModel._create_and_replace = _create_and_replace
    mixed.model.MixedModel._create_new_module = _create_new_module
    peft.PeftMixedModel.save_pretrained = save_pretrained

    # fix remove colom
    from transformers import Trainer
    import transformers
    Trainer._set_signature_columns_if_needed = _set_signature_columns_if_needed
    transformers.trainer.can_return_loss = can_return_loss
