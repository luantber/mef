from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
import logging
from pytorch_lightning.loggers import TensorBoardLogger
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class Model(LightningModule):

    def custom_train(self, batch_size: int , epochs: int,  dataset: Dataset, debug=False):
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        logger = TensorBoardLogger("tb_logs", name="my_model")

        # Initialize a trainer
        trainer = Trainer(
            gpus=1,
            max_epochs=epochs,
            enable_model_summary=False,
            enable_progress_bar=debug,
            logger=logger
            # progress_bar_refresh_rate=20,
        )

        # Train the model
        trainer.fit(self, train_loader)


    def custom_validation(self, batch_size: int , dataset: Dataset):
        val_loader = DataLoader(dataset,  batch_size, shuffle=False)

        # Initialize a trainer
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
