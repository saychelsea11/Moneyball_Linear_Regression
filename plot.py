import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wrangling import df_transform

def plot_hist(data,transform="None"):
  data = df_transform(data,transform)

  count = 1
  plt.figure(figsize=(16,8))
  for col in data.columns:
    if col == 'runs':
      pass
    else: 
      plt.subplot(2,3,count)
      plt.hist(data[col])
      plt.title(col)
      count = count + 1

  plt.figure()
  plt.hist(data['runs'])
  plt.title('runs')

def plot_scatter(data,transform="None"):
  
  data = df_transform(data,transform)
  print (data.corr(method="pearson").style.background_gradient())

  count = 1
  plt.figure(figsize=(16,8))
  for col in data.columns:
    if col == "runs":
      pass
    else:
      a, b = np.polyfit(data["runs"], data[col], 1)
      plt.subplot(2,3,count)
      plt.scatter(data["runs"],data[col])
      plt.plot(np.array(data['runs']),a*np.array(data['runs'])+b,alpha=0.3,color='red')
      plt.xlabel("runs")
      plt.ylabel(col)
      count = count + 1
  plt.show()
  