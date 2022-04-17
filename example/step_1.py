# Import Dataset and models
from dataset.mnist import Mnist
from models.linear import Linear
from models.cnn import Cnn

# import mef
import mef

# Create Dataset 
dataset = Mnist()

# Configure Settings of models to test
settings = {
    "linear_1": mef.Setting(Linear, batch_size=128, epochs=100),
    # "cnn_1": mef.Setting(Cnn, batch_size=128, epochs=100),
}

# Create experiment 
exp = mef.Experiment(settings=settings, dataset=dataset)

# 
results = exp.test("linear_1")
print(results)