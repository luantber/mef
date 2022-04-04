import torch
import torchmetrics 
from torch.nn import functional as F
from torch.nn import Sequential
from mef import Model 

class Cnn(Model):

    def __init__(self):
        super().__init__()
        self.l1 = Sequential(
            torch.nn.Conv2d(1,64,3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.Conv2d(64,128,3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(10),
        )
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return torch.relu(self.l1(x))

    def training_step(self, batch, idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("loss",loss)
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