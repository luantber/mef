from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import pickle
from faker import Faker
import os
fake = Faker()

@dataclass
class Iteration:

    number_kfolds: int
    idx: int
    path_save: str
    model_name: str = None
    set_name: str = None
    kfolds: list[dict] = field(default_factory=list)
    create_at: datetime = field(default_factory=lambda: str(datetime.now()))

    def store(self):

        path = os.path.join( self.path_save ,f"{self.set_name}_{self.model_name}")
        if not os.path.isdir(path):
            os.mkdir(path)
        
        file_path = os.path.join( path  , f"{self.set_name}={self.model_name}={self.idx}.pk" )
        with open( file_path , "wb" )as f:
            pickle.dump(self, f)

    def append(self, kfold_result):
        self.kfolds.append(kfold_result)


@dataclass
class IterationSet:
    model_name: str
    path_save: str
    experiment_name: str = field(default_factory=lambda : f'{fake.color_name()}{fake.first_name()}')
    iterations: list[Iteration] = field(default_factory=list)
    create_at: datetime = field(default_factory=lambda: str(datetime.now()))

    def store(self):
        path = os.path.join( self.path_save, f"{self.experiment_name}={self.model_name}.pk" )
        with open( path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str)->IterationSet:

        with open(path, "rb") as file:
            data = pickle.load(file)
            return data

    def append(self, iteration: Iteration):
        iteration.set_name = self.experiment_name
        iteration.model_name = self.model_name
        self.iterations.append(iteration)
        self.store()