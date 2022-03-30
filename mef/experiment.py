from typing import Any, Dict
from torch.utils.data import Dataset, Subset
import torch 
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.model_selection import KFold
from pytorch_lightning import LightningModule


@dataclass
class Experiment:
    models: Dict[str, LightningModule]
    dataset: Dataset
    seed: int = 42

    ## Single Step of the nested loops
    def train_single(self, model_name: str, dataset: Dataset):
        # Factory Reset Model
        torch.manual_seed(0)
        model = self.models[model_name]()
        model.custom_train(dataset)
        return model

    def validate_single(self, model: LightningModule, dataset: Dataset):
        results = model.custom_validation(dataset)
        return results

    def run_single(self, model:str, idx_iteration:int, kfold:int):
        """
            This run a single iteration
            the seed used to shuffle is the idx_iteration

            this should store an object that contains 

            - name_model
            - results  len(k fold )
            - idx iteration
        """
        print(f"\nIteration {idx_iteration}")

        fold_result = []

        kf = KFold(n_splits=kfold, shuffle=True, random_state=idx_iteration)
        for train_idx, test_idx in kf.split(self.dataset):

            print("fold")
            print(test_idx)

            train = Subset(self.dataset, train_idx)
            test = Subset(self.dataset, test_idx)

            model_trained = self.train_single(model, train)
            results = self.validate_single(model_trained, test)

            fold_result.append(results)
        
        return fold_result

    def run_model(self, model:str , iterations: int , kfold:int =4):
        """
            Can collect the results of single iteration
        """
        iterations_results = []
        for i in range(iterations):
            result = self.run_single(model, i, kfold)
            iterations_results.append(result)
        return iterations_results

    def run_all(self):
        """
            Can collect multiple runs 
        """
        print("all models")
