# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:06:40 2017

@author: Clayton

This program demonstrates how two different methods of regression analysis, 
regression trees and ordinary least squares regression compare and relate
in their prediction of total expenditures given first a single variable 
(income before taxes) and then a set of independent variables.

"""

import pandas as pd
import statsmodels.api as sm
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import pydotplus

def fmli_reader():
    #create an empty DataFrame and declare the location for the source files on github
    fmli = pd.DataFrame()
    url = 'https://raw.githubusercontent.com/clayknappen/CE-Trees-and-Linear-Regression/master/fmli'
    
    cols = ['TOTEXPPQ', 'FINCBTAX', 'REF_RACE', 'SEX_REF', 'EDUC_REF', 'AGE_REF', 'FINLWT21']
    dtypes = {'TOTEXPPQ' : float, 'FINCBTAX' : float, 'REF_RACE' : str, 
              'SEX_REF' : str, 'EDUC_REF' : str, 'AGE_REF' : int}
    
    for i in range(1,6):
        if i == 1:
            df = pd.read_csv(url + '151x.csv', usecols=cols, dtype=dtypes)
        elif i == 5:
            df = pd.read_csv(url + '161.csv', usecols=cols, dtype=dtypes)
        else:
            df = pd.read_csv(url + '15' + str(i) + '.csv', usecols=cols, dtype=dtypes)
        fmli = fmli.append(df)
        del df
        print i
    return fmli

fmli15 = fmli_reader()

X = fmli15['FINCBTAX']
X_const = sm.add_constant(X, prepend=False) #add an intercept term for Statsmodels to use in fitting OLS

y = fmli15['TOTEXPPQ'] * 4

ols = sm.OLS(y, X_const)

res = ols.fit()
print res.summary()

rt = DecisionTreeRegressor()

rt.fit(X_const, y)
print rt.score(X_const,y)

dot_data = tree.export_graphviz(rt, out_file=None,
                                feature_names=['FINCBTAX', 'CONST'],
                                filled=True)
 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("DecisionTreeRegressor.pdf")