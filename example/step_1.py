from dataset.mnist import Mnist
from pytorch_lightning import seed_everything
from models.linear import Linear
from models.cnn import Cnn

seed_everything(42, workers=True)
import mef
from mef import Setting

dataset = Mnist()

settings = {
        "linear_1": Setting(Linear, batch_size=128, epochs=100),
        "cnn_1": Setting(Cnn, batch_size=128, epochs=100),
}
    
exp = mef.Experiment(
    settings=settings, dataset=dataset
)

results = exp.test("linear_1")

print(results)