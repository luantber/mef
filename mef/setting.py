from dataclasses import dataclass
from mef import Model

@dataclass
class Setting:
    model: type[Model]
    batch_size: int
    epochs: int

    def create_model(self) -> Model:
        # Instantiate model
        return self.model()
