import context

from dataset.mnist import Mnist
from pytorch_lightning import seed_everything
from models.linear import Linear 
from models.cnn import Cnn

seed_everything(42, workers=True)
import mef

dataset = Mnist()
exp = mef.Experiment( models={"Linear": Cnn} , dataset=dataset, 
        batch_size=128, epochs=150 
)


results = exp.test("Linear")
print ( results )