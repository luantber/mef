
from models.linear import Linear ## To be improved
from dataset.mnist import Mnist
from torch.utils.data import Dataset


"""
    This will be hidden ( will be part of the library )
"""
class Experiment:
    def __init__(self):
        self.models = {
            "Linear": Linear,
        }

    def add_models(self,models: dict):
        self.models = models 

    def train_single(self,model_name: str ,dataset: Dataset ):
        model = self.models[model_name]()
        model.custom_train(dataset)

    def run(self, kfold = 4, iterations = 10 , metric = "accuracy" ):
        
        for i in range(iterations):
            print(f"Iteration {i}")
            for model_name in self.models:
                self.train_single(model_name, Mnist() )


####### ___________________________ ############# ___________---


if __name__ == "__main__":

    dataset = Mnist()

    first_exp = Experiment()
    first_exp.add_models({"Linear": Linear})  # or something similar
    first_exp.run( kfold = 4, iterations = 10 , metric = "accuracy" )
    







    

    

