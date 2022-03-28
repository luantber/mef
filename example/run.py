import context
from models.linear import Linear ## To be improved
from dataset.mnist import Mnist

# import library
import mef

if __name__ == "__main__":
    
    dataset = Mnist()

    

    """
        STEP 1

        Create Experiment with the following parameters: 
            - Dictionary of Models
            - Dataset to be used
            - Number of Iterations (move to run ? ) 
            - Metric(s) to evaluate
    """
    first_exp = mef.Experiment()
    first_exp.add_models({"Linear": Linear})  # or something similar


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

    first_exp.run( kfold = 4, iterations = 10 , metric = "accuracy" )
