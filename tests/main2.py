import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

print(os.path.dirname(__file__))
print(os.pardir)
print(PROJECT_ROOT)

import mef 
print ( mef.Experiment("juan") )