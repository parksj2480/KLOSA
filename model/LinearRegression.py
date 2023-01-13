#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Machine Learning 

# In[37]:


import pandas as pd
import numpy as np
import random
import os

import warnings
warnings.filterwarnings('ignore')


# In[23]:


# 일정한 결과를 위해 seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정 42


# In[44]:


from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import get_scorer_names
# Grid Search 함수 정의
def modelfit(model, grid_params, X_train, y_train, X_test, y_test): 
    model = GridSearchCV(estimator=model,
                        param_grid=grid_params,
                        cv=4,
                        refit=True,
                        scoring='neg_root_mean_squared_error'
                        )

    model.fit(X_train, y_train)
    print('Train Done.')

    #Predict training set:
    y_pred = model.predict(X_test)

    #Print model report:
    print("\nModel Report")
    #print("\nCV 결과 : ", model.cv_results_)
    print("\n베스트 정답률 : ", model.best_score_)
    print("\n베스트 파라미터 : ", model.best_params_)
    print("Test 점수 : ",np.sqrt(mean_squared_error(y_test, y_pred)))
    return model


# In[2]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[24]:


# Machine Learning 에서는 같은 사람이라도 설문 차수가 다르면 다른 데이터로 취급
df=pd.read_csv("../data/LongData.csv")


# In[25]:


df.head()


# In[5]:


# 제거 할 변수 추가 < region1, 낙상, adl ,iadl, 악력 > 제거

df=df[df.columns.drop(list(df.filter(regex=r'(region1|adl|C056)')))]


# In[6]:


df.shape


# In[7]:


df_drop=df.dropna(axis=0)


# In[9]:


X=df_drop.drop(['mmse','mmseg'],axis=1).iloc[:,1:]
y=df_drop['mmse']


# In[10]:


X=X.fillna(0)
y=y.fillna(0)


# In[11]:


print(X.shape,y.shape)


# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[13]:


print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[45]:


# LinearRegression Model

from sklearn.linear_model import LinearRegression

model=LinearRegression()
print("Model : ", model)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("기본 점수 :",np.sqrt(mean_squared_error(y_test, y_pred)))

grid_params={
    'normalize':[True,False]
}

model=modelfit(model, grid_params, X_train, y_train, X_test, y_test)


# In[46]:


# DecisionTree Regression Model

from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor()
print("Model : ", model)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("기본 점수 :",np.sqrt(mean_squared_error(y_test, y_pred)))

grid_params={
    'min_samples_leaf': [1, 2, 3],
    'max_depth': [1, 2, 3, 4, 5]
}

model = modelfit(model, grid_params, X_train, y_train, X_test, y_test)


# In[51]:


# RandomForest Regression Model
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor()
print("Model : ", model)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("기본 점수 :",np.sqrt(mean_squared_error(y_test, y_pred)))

grid_params={
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [1,2,3,4],
}

model = modelfit(model, grid_params, X_train, y_train, X_test, y_test)


# In[52]:


# XGboost Regression Model
import xgboost as xgb 

model = xgb.XGBRegressor()
print("Model : ", model)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("기본 점수 :",np.sqrt(mean_squared_error(y_test, y_pred)))

grid_params={
    "colsample_bytree":[1.0],
    "min_child_weight":[1.0,1.2],
    'max_depth': [3,4,6,7,8,9,10], 
    'n_estimators': [250,500,1000,1500,2000]
}

model = modelfit(model, grid_params, X_train, y_train, X_test, y_test)


# In[ ]:


# Support Vector Regression Model
from sklearn.svm import SVR

model=SVR()
print("Model : ", model)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("기본 점수 :",np.sqrt(mean_squared_error(y_test, y_pred)))

grid_params={
    'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
    'C' : [1,5,10],
    'degree' : [3,8],
    'coef0' : [0.01,10,0.5],
    'gamma' : ('auto','scale')
}

model = modelfit(model, grid_params, X_train, y_train, X_test, y_test)

