from models.linear import Linear  ## To be improved  to " from models import Linear "
from models.cnn import Cnn
from dataset.mnist import Mnist  ##< "from dataset import Mnist"

import warnings

warnings.filterwarnings("ignore")

import mef

dataset = Mnist()

settings = {
    "linear_1": mef.Setting(Linear, batch_size=256, epochs=25),
    "cnn_1": mef.Setting(Cnn, batch_size=256, epochs=25),
}

exp = mef.Experiment(settings=settings, dataset=dataset, path_save="logs/")

# 0,1,2,3,4,5,6,7,8,9

exp.run_model("linear_1", iterations_range=range(10), kfold=6)
exp.run_model("cnn_1", iterations_range=range(10), kfold=6)

