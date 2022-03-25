class AbstractModel:
    def __init__(self):
        print("Initilizing AbstractModel")

    def forward(self):
        pass


class NeuralNet(AbstractModel):
    def forward(self):
        print("hi Neural Net")


class ConvNet(AbstractModel):
    def forward(self):
        print("hi Conv Net")



def factoryModel(name):
    clases = {
        "ConvNet": ConvNet,
        "NeuralNet": NeuralNet
    }
    print(type(ConvNet))
    print(type(ConvNet()))
    return clases[name]()


if __name__ == '__main__':
    model = factoryModel('ConvNet')
    model.forward()

    model = factoryModel('ConvNet')
    model.forward()