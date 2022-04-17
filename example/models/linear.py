import torch
import torchmetrics
from torch.nn import functional as F
from mef import Model


class Linear(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        preds = self(x)
        self.accuracy(preds, y)

        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
