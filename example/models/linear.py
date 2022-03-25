import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer

class Linear(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def custom_train(self,dataset: Dataset ):
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize a trainer
        trainer = Trainer(
            gpus=1,
            max_epochs=3,
            progress_bar_refresh_rate=20,
        )

        # Train the model
        trainer.fit( self , train_loader)