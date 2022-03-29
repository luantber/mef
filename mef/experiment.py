from typing import Any, Dict
from torch.utils.data import Dataset, Subset

from dataclasses import dataclass
from typing import Dict, Any

from sklearn.model_selection import KFold

from pytorch_lightning import LightningModule


@dataclass
class Experiment:
    models: Dict[str, Any]
    dataset: Dataset
    seed: int = 42

    ## Single Step of the nested loops
    def train_single(self, model_name: str, dataset: Dataset):
        # Factory Reset Model
        model = self.models[model_name]()
        model.custom_train(dataset)
        
        # print(f"Single step: {model_name} , data {dataset[0]}")
        return model

    def test_single(self, model: LightningModule, dataset: Dataset):
        results = model(dataset)
        return results 

    def run(self, iterations, kfold=4, metric="accuracy"):
        # To replicate the experiments
        for i in range(iterations):

            print(f"\nIteration {i}")
            kf = KFold(n_splits=kfold, random_state=self.seed, shuffle=True)  ## is it ok having the same seed ???? 

            for train_idx, test_idx in kf.split(self.dataset):
                
                train = Subset(self.dataset,train_idx)
                test = Subset(self.dataset,test_idx)

                # For each model
                for model_name in self.models:
                    model_trained = self.train_single(model_name, train  )
                    results = self.test_single(model_trained, test  )