from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import pickle
from faker import Faker
import os
from numpy import mean
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

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

        path = os.path.join(self.path_save, f"{self.set_name}_{self.model_name}")
        if not os.path.isdir(path):
            os.mkdir(path)

        file_path = os.path.join(
            path, f"{self.set_name}={self.model_name}={self.idx}.pk"
        )
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def append(self, kfold_result):
        self.kfolds.append(kfold_result)


@dataclass
class IterationSet:
    model_name: str
    path_save: str
    experiment_name: str = field(
        default_factory=lambda: f"{fake.color_name()}{fake.first_name()}"
    )
    iterations: list[Iteration] = field(default_factory=list)
    create_at: datetime = field(default_factory=lambda: str(datetime.now()))

    def store(self):
        path = os.path.join(
            self.path_save, f"{self.experiment_name}={self.model_name}.pk"
        )
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> IterationSet:

        with open(path, "rb") as file:
            data = pickle.load(file)
            return data

    def append(self, iteration: Iteration):
        iteration.set_name = self.experiment_name
        iteration.model_name = self.model_name
        self.iterations.append(iteration)
        self.store()


def show_results(it_a: IterationSet, it_b: IterationSet):
    get_ids_func = lambda it: it.idx
    idx_a = set(map(get_ids_func, it_a.iterations))
    idx_b = set(map(get_ids_func, it_b.iterations))

    it_a.iterations.sort(key=get_ids_func)
    it_b.iterations.sort(key=get_ids_func)

    common_idx = list(idx_a & idx_b)

    is_in_common_func = lambda it: it.idx in common_idx

    new_its_a = list(filter(is_in_common_func, it_a.iterations))
    new_its_b = list(filter(is_in_common_func, it_b.iterations))

    assert len(new_its_a) == len(new_its_b)
    print(common_idx)

    ps = []
    ws = []
    means = []

    for i in range(len(new_its_a)):
        iteration_a = new_its_a[i]
        iteration_b = new_its_b[i]

        iteration_a_acc = list(map(lambda i: i[0]["val_acc"], iteration_a.kfolds))
        iteration_b_acc = list(map(lambda i: i[0]["val_acc"], iteration_b.kfolds))

        # print(iteration_a.kfolds)
        # print(iteration_b.kfolds)
        mean_a = mean(iteration_a_acc)
        mean_b = mean(iteration_b_acc)
        print(iteration_a_acc, mean_a)
        print(iteration_b_acc, mean_b)
        w, p = wilcoxon(iteration_a_acc, iteration_b_acc)
        mean_ = abs(mean_b - mean_a)
        print(w, p, mean_ )

        # print("\n\n")
        ps.append(p)
        ws.append(w)
        means.append(mean_)


    plt.hist(ws, bins=range(10), ec="black",density=True)
    plt.axvline(1, color="k", linestyle="dashed", linewidth=1)
    plt.xticks(range(10))
    plt.show()




    # fig, ax = plt.subplots()

    # sns.histplot(ws, ax=ax)
    # sns.histplot(ps, binwidth=0.05, ax=ax)
    # ax.set_xlim(0, 1.5)
    # ax.set_xticks([0.05,0.1,0.5,1])
    # plt.show()
