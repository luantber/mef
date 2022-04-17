from dataclasses import dataclass
from mef import Model


@dataclass
class Setting:
    model_type: type[Model]
    batch_size: int
    epochs: int

    def create_model(self, model_name: str) -> Model:
        # Instantiate model
        ins = self.model_type(model_name)
        return ins
