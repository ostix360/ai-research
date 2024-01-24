import warnings
from typing import List, Optional

import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose

from .layer import fffLayer

if is_bnb_available():
    import bitsandbytes as bnb


    class Linear8bitLt(torch.nn.Module, fffLayer):

        def __init__(
                self,
                base_layer: torch.nn.Module,
                adapter_name: str,
                intermediate_size: int,
                num_fff: int = 1,
                activation_func: str = "silu",
                init_fff_weights: bool = True,
                **kwargs,
        ):
            super().__init__()
            fffLayer.__init__(self, base_layer, **kwargs)
            self.update_layer(adapter_name, intermediate_size, num_fff, activation_func, init_fff_weights)

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                result = self.base_layer(x, *args, **kwargs)
            else:
                hidden_states = x
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.fff.keys():
                        continue

                    _fff = self.fff[active_adapter]
                    _fff_layer_norm = self.fff_layer_norms[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = self.base_layer.dtype
                        compute_dtype = _fff[0].u_proj.weight.dtype
                        if hidden_states.dtype != compute_dtype:
                            hidden_states = hidden_states.to(compute_dtype)
                    for fff, fff_layernorm in zip(_fff, _fff_layer_norm):
                        residual = hidden_states
                        hidden_states = fff_layernorm(hidden_states)
                        hidden_states = fff(hidden_states)
                        hidden_states = residual + hidden_states
                    if requires_conversion:
                        hidden_states = hidden_states.to(expected_dtype)
                result = self.base_layer(hidden_states, *args, **kwargs)
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "fff." + rep

if is_bnb_4bit_available():
    import bitsandbytes as bnb

    class Linear4bit(torch.nn.Module, fffLayer):
        def __init__(
                self,
                base_layer: torch.nn.Module,
                adapter_name: str,
                intermediate_size: int,
                num_fff: int = 1,
                activation_func: str = "silu",
                init_fff_weights: bool = True,
                **kwargs,
        ):
            super().__init__()
            fffLayer.__init__(self, base_layer, **kwargs)
            self.update_layer(adapter_name, intermediate_size, num_fff, activation_func, init_fff_weights)

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                result = self.base_layer(x, *args, **kwargs)
            else:
                hidden_states = x
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.fff.keys():
                        continue

                    _fff = self.fff[active_adapter]
                    _fff_layer_norm = self.fff_layer_norms[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = hidden_states.dtype
                        hidden_states = hidden_states.to(_fff[0].u_proj.weight.dtype)
                    for fff, fff_layernorm in zip(_fff, _fff_layer_norm):
                        residual = hidden_states
                        hidden_states = fff_layernorm(hidden_states)
                        hidden_states = fff(hidden_states)
                        hidden_states = residual + hidden_states
                    if requires_conversion:
                        hidden_states = hidden_states.to(expected_dtype)
                result = self.base_layer(hidden_states, *args, **kwargs)
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "fff." + rep
