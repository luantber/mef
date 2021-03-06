from typing import Optional
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from dataclasses import dataclass
from sklearn.model_selection import KFold

from mef.iteration import Iteration, IterationSet
from mef.model import Model
from mef.setting import Setting

import pytorch_lightning as pl


@dataclass
class Experiment:
    settings: dict[str, Setting]
    dataset: Dataset
    seed: int = 42
    path_save: str = "logs/"

    def __post_init__(self):
        if not os.path.isdir(self.path_save):
            os.makedirs(self.path_save)

    def train_single(
        self,
        setting_id: str,
        dataset: Dataset,
        debug=False,
        val_dataset: Dataset = None,
    ) -> Model:

        """
            This function train a model using the dataset, returns a model.
        """

        pl.utilities.seed.seed_everything(seed=0, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        setting = self.settings[setting_id]

        model: Model = setting.create_model(setting_id)
        model.custom_train(
            setting.batch_size,
            setting.epochs,
            dataset,
            debug=debug,
            val_dataset=val_dataset,
            dataloader_args=setting.dataloader_args,
        )
        return model

    def validate_single(self, model: Model, setting_id: str, dataset: Dataset):
        """
            This function validates the model using the dataset
        """
        pl.utilities.seed.seed_everything(seed=0, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        setting = self.settings[setting_id]
        results = model.custom_validation(
            setting.batch_size, dataset, dataloader_args=setting.dataloader_args
        )
        return results

    def run_single(self, setting_id: str, idx_iteration: int, kfold: int) -> Iteration:
        """
            This run a single iteration
            the seed used to shuffle is the idx_iteration

            Stores an object (Iteration) that contains 

            - name_model
            - results  len(k fold )
            - idx iteration
        """

        print(f"\nIteration {idx_iteration}")

        # kf_iteration = Iteration( kfold, setting_id, idx_iteration, self.path_save)
        kf_iteration = Iteration(kfold, idx_iteration, self.path_save)

        kf = KFold(n_splits=kfold, shuffle=True, random_state=idx_iteration)
        for train_idx, test_idx in tqdm(
            kf.split(self.dataset), total=kf.get_n_splits(), desc="k-fold"
        ):

            train = Subset(self.dataset, train_idx)
            test = Subset(self.dataset, test_idx)

            model_trained = self.train_single(setting_id, train)
            results = self.validate_single(model_trained, setting_id, test)

            kf_iteration.append(results)

        return kf_iteration

    def run_model(
        self, setting_id: str, iterations_range: range, kfold: int = 4
    ) -> IterationSet:
        """
            Collects the results of single iterations
        """
        iterations_set = IterationSet(setting_id, self.path_save)

        for i in iterations_range:
            kf_iteration = self.run_single(setting_id, i, kfold)
            iterations_set.append(kf_iteration)
            kf_iteration.store()

        return iterations_set

    def test(self, setting_id: str, seed: Optional[int] = None):
        """
        Simulates a single iteration of training of validation
        """

        size_train = int(len(self.dataset) * 0.75)
        size_test = len(self.dataset) - size_train

        train, test = torch.utils.data.random_split(
            self.dataset,
            [size_train, size_test],
            generator=torch.Generator().manual_seed(seed) if seed else None,
        )

        model = self.train_single(setting_id, train, debug=True, val_dataset=test)
        result = self.validate_single(model, setting_id, test)
        return result

    # def run_all(self):
    #     """
    #         Can collect multiple runs
    #     """
    #     print("all models")
