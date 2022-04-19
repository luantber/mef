## New to do

- Script to explore IterationSet.pk

```
  python myscrip.py algo.pk

  show info of .pk in console
```

- Bug `def custom_validation` in Model, returns a list , should be a dictionary

# Old

## Cosas por hacer

- Como recuperar la informacion de los archivos .pk
- Meter todos los archivos en una carpeta
- Como identificar los archivos, metadata ???

```
 # Create Dataset

  # my_dataset = Mnist()
```

## STEP 1

Create Experiment with the following parameters:

- Dictionary of Models (X)
- Dataset to be used (X)
- Number of Iterations (move to run ? ) (X)
- Metric(s) to evaluate ( )
  """

```
  # first_exp = mef.Experiment(models={"Linear": Linear, "CNN": Linear}, dataset=my_dataset, batch_size=128, epochs=2 )
```

## STEP 2

Run the experiment
which is a nested fors

- Iterations
- K Fold
- Models using same data
  Should store "metric" of evaluated.
  Should allow resume ( How to know if )
  Reproducible ( fixed seeds )

```
  # first_exp.run_model("Linear", n_iterations=2,kfold=4)
```
