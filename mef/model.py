import logging
from statistics import mode
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from dataclasses import dataclass

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)



class Model(LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def custom_train(
        self,
        batch_size: int,
        epochs: int,
        dataset: Dataset,
        debug=False,
        val_dataset: Optional[Dataset] = None,
    ):
        train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if val_dataset
            else None
        )

        logger = TensorBoardLogger("tb_logs", name=self.model_name)

        trainer = Trainer(
            gpus=1,
            max_epochs=epochs,
            enable_model_summary=False,
            enable_progress_bar=debug,
            logger=logger
            # progress_bar_refresh_rate=20,
        )

        # Train the model
        trainer.fit(self, train_loader, val_dataloaders=val_loader)

    def custom_validation(self, batch_size: int, dataset: Dataset):
        val_loader = DataLoader(dataset, batch_size, shuffle=False)

        trainer = Trainer(
            gpus=1,
            enable_model_summary=False,
            enable_progress_bar=False
            # max_epochs=3,
            # progress_bar_refresh_rate=20,
        )

        # First position
        result = trainer.validate(self, val_loader, verbose=False)
        return result
