from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import pickle
from faker import Faker
import os
fake = Faker()

@dataclass
class Iteration:

    number: int
    idx: int
    set_name: str
    model_name: str = None
    kfolds: list[dict] = field(default_factory=list)
    create_at: datetime = field(default_factory=lambda: str(datetime.now()))

    def store(self):

        path = f"{self.set_name}_{self.model_name}"
        if not os.path.isdir(path):
            os.mkdir(path)
        

        with open( os.path.join( path  , f"{self.set_name}={self.model_name}={self.idx}.pk", "wb") )as f:
            pickle.dump(self, f)

    def append(self, kfold_result):
        self.kfolds.append(kfold_result)


@dataclass
class IterationSet:
    model_name: str
    experiment_name: str = field(default_factory=lambda : f'{fake.color_name()}{fake.first_name()}')
    iterations: list[Iteration] = field(default_factory=list)
    create_at: datetime = field(default_factory=lambda: str(datetime.now()))

    def store(self):
        with open(f"{self.experiment_name}={self.model_name}.pk", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str)->IterationSet:

        with open(path, "rb") as file:
            data = pickle.load(file)
            return data

    def append(self, iteration: Iteration):
        iteration.set_name = self.experiment_name
        self.iterations.append(iteration)
        self.store()