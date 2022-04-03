import torch
import torchmetrics 
from torch.nn import functional as F
from torch.nn import Sequential
from mef import Model 

class Cnn(Model):

    def __init__(self):
        super().__init__()
        self.l1 = Sequential(
            torch.nn.Conv2d(1,10,3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(),
            torch.nn.Linear(28 * 28, 10),
        )
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        preds = self(x)
        self.accuracy(preds, y)
                
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)