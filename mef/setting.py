from dataclasses import dataclass, field
from typing import Optional, Any
from mef import Model


@dataclass
class Setting:
    model_type: type[Model]
    batch_size: int
    epochs: int
    dataloader_args: dict[str, Any] = field(default_factory=dict)

    def create_model(self, model_name: str) -> Model:
        # Instantiate model
        ins = self.model_type(model_name)
        return ins
