import context
from models.linear import Linear  ## To be improved  to " from models import Linear "
from dataset.mnist import Mnist  ##< "from dataset import Mnist"

from pytorch_lightning import seed_everything
import warnings
warnings.filterwarnings('ignore')

seed_everything(42, workers=True)

import mef

if __name__ == "__main__":

    # Create Dataset
    my_dataset = Mnist()

    """
        STEP 1

        Create Experiment with the following parameters: 
            - Dictionary of Models                       (X)
            - Dataset to be used                         (X)
            - Number of Iterations (move to run ? )      (X) 
            - Metric(s) to evaluate                      ( )
    """

    first_exp = mef.Experiment(models={"Linear": Linear, "CNN": Linear}, dataset=my_dataset)

    """
        STEP 2

        Run the experiment 
        which is a nested fors 
            - Iterations
            - K Fold 
            - Models using same data

        Should store "metric" of evaluated. 
        Should allow resume   (  How to know if )
        Reproducible ( fixed  seeds )
    """


    first_exp.run_model("Linear", iterations=2,kfold=4)


