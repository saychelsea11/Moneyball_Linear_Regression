import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import *

def simple_linear_regression(df,x,y):
    eq = y + ' ~ ' + x
    m = ols(eq,df).fit()
    print ("Equation:",eq,'\n')
    print(m.summary())
    return m
    
def multiple_linear_regression(df,x_vars,y):
    eq = y + ' ~ '
    for i in x_vars:
        eq = eq + i + ' + '
    eq = eq[:-3]
    print ("Equation:",eq,'\n')
    m = ols(eq,df).fit()
    print(m.summary())
    return m
    
def model_diagnostics(m):
    #Printing residuals vs fitted values
    #a, b = np.polyfit(m.resid, m.fittedvalues, 1)
    plt.figure()
    sns.regplot(x=m.fittedvalues,y=m.resid,fit_reg=True)
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.show()