import context

from dataset.mnist import Mnist
from pytorch_lightning import seed_everything
from models.linear import Linear 
seed_everything(42, workers=True)
import mef

dataset = Mnist()
exp = mef.Experiment( models={"Linear": Linear} , dataset=dataset, 
        batch_size=128, epochs=10 
)


results = exp.test("Linear")
print ( results )