from scipy.__config__ import show
from mef.iteration import IterationSet, show_results

# cnn_set = IterationSet.load("logs/DarkVioletCraig=cnn_1.pk")
# linear_set = IterationSet.load("logs/OrangeRedZachary=linear_1.pk")
# cnn_set = IterationSet.load("logs/NavajoWhiteWilliam=cnn_1.pk")
# linear_set = IterationSet.load("logs/LavenderMichelle=linear_1.pk")

cnn_set = IterationSet.load("logs/ForestGreenKimberly=cnn_1.pk")
linear_set = IterationSet.load("logs/LimeAaron=linear_1.pk")

show_results(cnn_set, linear_set)

