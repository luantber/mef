from torch.utils.data import Dataset
class Dummy(Dataset):
    def __init__(self):
        self.data = range(0,40)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return f's_{self.data[index]}'
