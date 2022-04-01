from dataclasses import dataclass, field
import pickle


@dataclass
class Iteration:

    number: int
    model_name: str
    idx: int
    kfolds: list[dict] = field(default_factory=list)

    def store(self):
        with open(f'{self.model_name}_{self.idx}.pk', 'wb') as f:
            pickle.dump(self, f)

    def append(self, kfold_result ):
        self.kfolds.append( kfold_result )


@dataclass
class IterationSet:
    model_name: str
    iterations: list[Iteration] = field(default_factory=list)


    def store(self):
        with open(f'set_{self.model_name}.pk', 'wb') as f:
            pickle.dump(self, f)

    def append(self, iteration : Iteration):
        self.iterations.append(iteration)
        self.store()


