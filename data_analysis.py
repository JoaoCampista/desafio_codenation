#%% Bibliotecas

import pandas as pd
import numpy as np
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression


from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor


#%% Dataset

dataset_test = pd.read_csv(r'C:\Users\joao\OneDrive\Road To Developer\desafio-codenation\dataset\test.csv',
                           sep =',',
                           low_memory = False)

dataset_train = pd.read_csv(r'C:\Users\joao\OneDrive\Road To Developer\desafio-codenation\dataset\train.csv',
                            sep =',',
                            low_memory = False)

#%% Colunas avaliadas

dataset_dict = list(pd.read_csv(r'C:\Users\joao\OneDrive\Road To Developer\desafio-codenation\dataset\colunas_avaliadas.csv',
                                sep =',',
                                low_memory = False)["colunas_avaliadas"])

    
#%% Limpando os dados de treino

dataset_train = dataset_train[dataset_dict]



#%% Tratando os dados

for index, row in enumerate(dataset_train["NU_NOTA_CN", "NU_NOTA"]):
    if (row >= 1) & (row <250):
        dataset_train["NU_NOTA_CN"][index] = 1
    
    if (row >= 250) & (row < 500):
        dataset_train["NU_NOTA_CN"][index] = 2
        
    if (row >= 500) & (row <750):
        dataset_train["NU_NOTA_CN"][index] = 3
        
    if (row >= 750) & (row <1000):
        dataset_train["NU_NOTA_CN"][index] = 4

#%%
substituir_questoes = {
    "A" : 1,
    "B" : 2,
    "C" : 3,
    "D" : 4,
    "E" : 5,
    "F" : 6,
    "G" : 7,
    "H" : 8,
    "I" : 9,
    "J" : 10,
    "K" : 11,
    "L" : 12,
    "M" : 13,
    "N" : 14,
    "O" : 15,
    "P" : 16,
    "Q" : 17,
}

substituir_sexo = {
    "F" : 0,
    "M" : 1,
}

dataset_train["Q006"] = dataset_train["Q006"].map(substituir_questoes)
dataset_train["Q024"] = dataset_train["Q024"].map(substituir_questoes)
dataset_train["Q025"] = dataset_train["Q025"].map(substituir_questoes)

dataset_train["TP_SEXO"] = dataset_train["TP_SEXO"].map(substituir_sexo)

dataset_train.fillna(0, inplace = True)

#%% separando os dados de treino e teste

X = dataset_train.iloc[:,0:24].values
y = dataset_train.iloc[:,24].values

#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#%% fit models

modelo = DecisionTreeRegressor()

modelo.fit(X_train, y_train)


#%% predicting test

y_pred = modelo.predict(X_test) 
    

cm = confusion_matrix(y_test, y_pred) 


#%%
accuracy = accuracy_score(y_test, y_pred, normalize = True)

#%% predicting test

y_pred = model.predict(X_test) 

cm = confusion_matrix(y_test, y_pred) 

accuracy = accuracy_score(y_test, y_pred, normalize = True)
