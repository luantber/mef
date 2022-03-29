from typing import Any, Dict
from torch.utils.data import Dataset

from dataclasses import dataclass
from typing import Dict, Any

# from sklearn.model_selection import KFold

@dataclass
class Experiment:
    models: Dict[str, Any]
    dataset: Dataset
    seed:int = 42

    ## Single Step of the nested loops 
    def train_single(self, model_name: str, dataset: Dataset):
        # Factory Reset Model
        model = self.models[model_name]()
        model.custom_train(dataset)

    def run(self,  iterations, kfold=4 , metric="accuracy"):
        # To replicate the experiments
        for i in range(iterations):

            print(f"Iteration {i}")
            # kf = KFold(n_splits=2,random_state=self.seed, shuffle=False)
            


            """
            Aqui programar el KFold dataset 
            for train,test in dataset:
                ## For each model
                for model_name in self.models:
                    self.train_single(model_name, Mnist() , TRAIN )
                    self.test_single(model_name, Mnist() , TEST )
            """

            # For each model
            for model_name in self.models:
                self.train_single(model_name, self.dataset)

