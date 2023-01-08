#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import *


# # Loading dataset

# In[ ]:


df = pd.read_csv(r'C:\Users\sdas\Github_DS\Moneyball_Linear_Regression\mlb11.csv',index_col=0)
print (df.head(),"\n")


# In[ ]:


print (df.describe(),"\n")


# # Correlation

# ### Plot correlation matrix

# In[ ]:


print (df.corr(method="pearson").style.background_gradient(),"\n")


# ### Drop variables with low correlation to *runs*

# In[ ]:


df = df.drop(['team','at_bats','strikeouts','stolen_bases','wins'],axis=1)
print (df.head(),"\n")


# # EDA

# ### Defining functions

# In[ ]:


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
  #plt.show()

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
  #plt.show()


# ### Plot histograms - *default* data

# In[ ]:


plot_hist(df)


# ### Plot histogram - *log* data

# In[ ]:


plot_hist(df.copy(),'log')


# ### Plot histogram - *square root* data

# In[ ]:


plot_hist(df.copy(),"root")


# Plot histogram - *square* data

# In[ ]:


plot_hist(df.copy(),"square")


# ### Plot scatter - *default* data

# In[ ]:


plot_scatter(df.copy())
print (df.corr(method="pearson").style.background_gradient(),"\n")


# ### Plot scatter - *log* data

# In[ ]:


plot_scatter(df.copy(),"log")
df_log = df_transform(df.copy(),"log")
print (df_log.corr(method="pearson").style.background_gradient(),"\n")


# ### Plot scatter - *square root* data

# In[ ]:


plot_scatter(df.copy(),"root")
df_root = df_transform(df.copy(),"root")
print (df_root.corr(method="pearson").style.background_gradient(),"\n")


# ### Plot scatter - *square* data

# In[ ]:


plot_scatter(df.copy(),"square")
df_square = df_transform(df.copy(),"square")
print (df_square.corr(method="pearson").style.background_gradient(),"\n")


##Simple linear regression
# statsmodels works nicely with pandas dataframes
# The thing inside the "quotes" is called a formula, a bit on that below
m = ols('runs ~ hits',df).fit()
print(m.summary())
m = ols('runs ~ homeruns',df).fit()
print(m.summary())
m = ols('runs ~ bat_avg',df).fit()
print(m.summary())
m = ols('runs ~ new_onbase',df).fit()
print(m.summary())
m = ols('runs ~ new_slug',df).fit()
print(m.summary())
m = ols('runs ~ new_obs',df).fit()
print(m.summary())
print ()
print (dir(m))
print ()
print ("Residuals:",m.resid)
print ("Fitted values:",m.fittedvalues)

#Printing residuals vs fitted values
plt.figure()
plt.scatter(m.fittedvalues,m.resid)
plt.show()



