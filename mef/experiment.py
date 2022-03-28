from torch.utils.data import Dataset

from dataclasses import dataclass


# @dataclass
# class Experiment:
#     name: str

class Experiment:
    def __init__(self):
        self.models = {
            # "Linear": Linear,
        }

    def add_models(self,models: dict):
        self.models = models 

    def train_single(self,model_name: str ,dataset: Dataset ):

        # Factory Reset Model
        model = self.models[model_name]()
        model.custom_train(dataset)

    def run(self, kfold = 4, iterations = 10 , metric = "accuracy" ):        
        
        # To replicate the experiments 
        seed = 42 

        for i in range(iterations):   
            print(f"Iteration {i}")


            """
            Aqui programar el KFold dataset 

            for train,test in dataset:
                
                ## For each model
                for model_name in self.models:
                    self.train_single(model_name, Mnist() , TRAIN )
                    self.test_single(model_name, Mnist() , TEST )

            """

            ## For each model
            for model_name in self.models:
                self.train_single(model_name, Mnist() )

