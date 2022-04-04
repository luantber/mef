import torch
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from dataclasses import dataclass
from sklearn.model_selection import KFold
from pytorch_lightning import LightningModule
from mef.iteration import Iteration, IterationSet
from mef.model import Model
from typing import Type 

@dataclass
class Experiment:
    models: dict[str, Type[Model] ]
    dataset: Dataset
    batch_size: int 
    epochs: int
    seed: int = 42

    ## Single Step of the nested loops
    def train_single(self, model_name: str, dataset: Dataset, debug=False, val_dataset: Dataset =None):
        # Factory Reset Model
        torch.manual_seed(0)
        model: Model = self.models[model_name]()
        model.custom_train(self.batch_size,self.epochs,dataset, debug=debug, val_dataset=val_dataset)
        return model

    def validate_single(self, model: Model, dataset: Dataset):
        results = model.custom_validation(self.batch_size, dataset)
        return results

    def run_single(self, model_name:str, idx_iteration:int, kfold:int) -> Iteration:
        """
            This run a single iteration
            the seed used to shuffle is the idx_iteration

            this should store an object that contains 

            - name_model
            - results  len(k fold )
            - idx iteration
        """
        print(f"\nIteration {idx_iteration}")

        
        kf_iteration = Iteration(kfold, model_name, idx_iteration)

        kf = KFold(n_splits=kfold, shuffle=True, random_state=idx_iteration)
        for train_idx, test_idx in tqdm(kf.split(self.dataset), total=kf.get_n_splits(), desc="k-fold"):

            train = Subset(self.dataset, train_idx)
            test = Subset(self.dataset, test_idx)

            model_trained = self.train_single(model_name, train)
            results = self.validate_single(model_trained, test)

            kf_iteration.append(results)
        
        return kf_iteration


    def run_model(self, model_name:str , iterations: int , kfold:int =4) -> IterationSet:
        """
            Can collect the results of single iteration
        """
        iterations_set = IterationSet(model_name)

        for i in range(iterations):
            kf_iteration = self.run_single(model_name, i, kfold)
            kf_iteration.store()
            iterations_set.append(kf_iteration)
            
        return iterations_set


    def test(self, model_name:str):
        size_train = int(len(self.dataset)*0.75)
        size_test = len(self.dataset) - size_train

        train, test = torch.utils.data.random_split(self.dataset, [ size_train , size_test ])

        model = self.train_single(model_name,train,debug=True, val_dataset = test )
        result = self.validate_single(model,test)
        return result

    # def run_all(self):
    #     """
    #         Can collect multiple runs 
    #     """
    #     print("all models")
