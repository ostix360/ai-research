from typing import Union, Optional, List, Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from transformers.activations import ACT2FN


# copied from MistralMLP and renamed to avoid confusion
class FFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.g_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.u_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.d_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.d_proj(self.act_fn(self.g_proj(x)) * self.u_proj(x))

# copied from MistralRMSNorm
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class fffLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("fff", "fff_layer_norms")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("intermediate_size", "num_fff", "activation_func")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.intermediate_size = {}
        self.num_fff = {}
        self.activation_func = {}
        self.fff = nn.ModuleDict({})
        self.fff_layer_norms = nn.ModuleDict({})

        self._disable_adapters = False
        self.kwarg = kwargs

        base_layer = self.get_base_layer()

        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
        else:
            raise NotImplementedError(f"Base layer {base_layer.__class__.__name__} not supported")
        self.in_features = in_features

    def update_layer(self, adapter_name, intermediate_size, num_fff, activation_func, init_fff_weights):
        if num_fff <= 0:
            raise ValueError("Number of FFN layers must be greater than 0")
        if len(intermediate_size) != num_fff:
            raise ValueError("Number of intermediate sizes must be equal to the number of FFN layers")
        if len(activation_func) != num_fff:
            raise ValueError("Number of activation functions must be equal to the number of FFN layers")
        for i in range(num_fff):
            if intermediate_size[i] <= 0:
                raise ValueError("Intermediate size must be greater than 0")
            if activation_func[i] not in ACT2FN.keys():
                raise ValueError(f"Activation function {activation_func} not supported"
                             f"Choose from {ACT2FN.keys()}")
        self.intermediate_size[adapter_name] = intermediate_size
        self.num_fff[adapter_name] = num_fff
        self.activation_func[adapter_name] = activation_func

        self.fff[adapter_name] = nn.ModuleList(
            [FFN(self.in_features, self.intermediate_size[adapter_name][i], self.activation_func[adapter_name][i]) for i in range(self.num_fff[adapter_name])]
        )
        self.fff_layer_norms[adapter_name] = nn.ModuleList(
            [RMSNorm(self.in_features) for _ in range(self.num_fff[adapter_name])]
        )
        self.reset_fff_parameters(adapter_name, init_fff_weights)

        weight = getattr(self.base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)


    def reset_fff_parameters(self, adapter_name, init_fff_weights):
        if init_fff_weights is False:
            return

        if adapter_name in self.fff.keys():
            for fff in self.fff[adapter_name]:
                fff.g_proj.weight.data.normal_(mean=0.0, std=0.02)
                fff.u_proj.weight.data.normal_(mean=0.0, std=0.02)
                fff.d_proj.weight.data.normal_(mean=0.0, std=0.02)
                if fff.g_proj.bias is not None:
                    fff.g_proj.bias.data.zero_()
                if fff.u_proj.bias is not None:
                    fff.u_proj.bias.data.zero_()
                if fff.d_proj.bias is not None:
                    fff.d_proj.bias.data.zero_()

            for fff_layernorm in self.fff_layer_norms[adapter_name]:
                fff_layernorm.weight.data.fill_(1.0)


class Linear(nn.Module, fffLayer):
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            intermediate_size: int = 0,
            num_fff: int = 1,
            activation_func: str = "silu",
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            init_fff_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        fffLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, intermediate_size, num_fff, activation_func, init_fff_weights)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            hidden_states = x
            for active_adapter in self.active_adapters:
                if active_adapter not in self.fff.keys():
                    continue
                _fff = self.fff[active_adapter]
                _fff_layer_norm = self.fff_layer_norms[active_adapter]
                hidden_states = hidden_states.to(_fff[0].u_proj.weight.dtype)
                for fff, fff_layernorm in zip(_fff, _fff_layer_norm):
                    residual = hidden_states
                    hidden_states = fff_layernorm(hidden_states)
                    hidden_states = fff(hidden_states)
                    hidden_states = residual + hidden_states
            hidden_states = hidden_states.to(previous_dtype)
            result = self.base_layer(hidden_states, *args, **kwargs)
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fff." + rep
