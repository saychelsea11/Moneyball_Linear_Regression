import pandas as pd
import numpy as np

def df_transform(data,transform="None"):
  for col in data.columns:
    if transform == 'log':
      data[col] = list(map(np.log,data[col]))
    elif transform == 'square':
      data[col] = list(map(np.square,data[col]))
    elif transform == 'root':
      data[col] = list(map(np.sqrt,data[col]))
    else: 
      pass
  
  return data
 