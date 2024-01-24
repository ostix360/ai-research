from dataclasses import dataclass, field
from typing import Union, Optional, List

from peft import PeftConfig, PeftType


@dataclass
class fffConfig(PeftConfig):
    target_modules: Union[List[str], str] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    intermediate_size: Union[List[int], int] = field(
        default=512,
        metadata={"help": "Intermediate size of the FFN layers"},
    )
    num_fff: int = field(
        default=1,
        metadata={"help": "Number of FFN layers"},
    )
    activation_func: Union[List[str], str] = field(
        default="gelu",
        metadata={"help": "Activation function of the FFN layers"},
    )
    init_fff_weights: bool = field(
        default=True,
        metadata={"help": "Initialize the weights of the FFN layers"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.FFF
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.intermediate_size = (
            [self.intermediate_size for _ in range(self.num_fff)] if isinstance(self.intermediate_size, int) else self.intermediate_size
        )
        self.activation_func = (
            [self.activation_func for _ in range(self.num_fff)] if isinstance(self.activation_func, str) else self.activation_func
        )
