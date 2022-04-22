from scipy.__config__ import show
from mef.iteration import IterationSet, show_results


linear_set = IterationSet.load("logs/DarkTurquoiseKimberly=linear_1.pk")
cnn_set = IterationSet.load("logs/LawnGreenLuis=cnn_1.pk")

show_results(cnn_set, linear_set)

