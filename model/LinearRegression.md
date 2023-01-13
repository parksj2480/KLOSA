# Linear Regression Machine Learning 


```python
import pandas as pd
import numpy as np
import random
import os

import warnings
warnings.filterwarnings('ignore')
```


```python
# 일정한 결과를 위해 seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정 42
```


```python
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
```


```python
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
```


```python
# Machine Learning 에서는 같은 사람이라도 설문 차수가 다르면 다른 데이터로 취급
df=pd.read_csv("../data/LongData.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>hhsize</th>
      <th>A002_age</th>
      <th>edu</th>
      <th>gender1</th>
      <th>region1</th>
      <th>region3</th>
      <th>C152</th>
      <th>C056</th>
      <th>C075</th>
      <th>C083</th>
      <th>C105</th>
      <th>C107</th>
      <th>C108</th>
      <th>C144</th>
      <th>bmi</th>
      <th>smoke</th>
      <th>iadl</th>
      <th>mmse</th>
      <th>mmseg</th>
      <th>chronic_a</th>
      <th>chronic_b</th>
      <th>chronic_c</th>
      <th>chronic_d</th>
      <th>chronic_e</th>
      <th>chronic_f</th>
      <th>chronic_g</th>
      <th>chronic_h</th>
      <th>chronic_i</th>
      <th>present_labor</th>
      <th>alc</th>
      <th>adl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>73.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>60.0</td>
      <td>152.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>25.969529</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.0</td>
      <td>51.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>59.0</td>
      <td>158.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>23.634033</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2.0</td>
      <td>52.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>72.0</td>
      <td>168.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>25.510204</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.0</td>
      <td>80.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>43.0</td>
      <td>143.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>21.027923</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>50.0</td>
      <td>157.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>20.284799</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 제거 할 변수 추가 < region1, 낙상, adl ,iadl, 악력 > 제거

df=df[df.columns.drop(list(df.filter(regex=r'(region1|adl|C056)')))]
```


```python
df.shape
```




    (88736, 28)




```python
df_drop=df.dropna(axis=0)
```


```python
X=df_drop.drop(['mmse','mmseg'],axis=1).iloc[:,1:]
y=df_drop['mmse']
```


```python
X=X.fillna(0)
y=y.fillna(0)
```


```python
print(X.shape,y.shape)
```

    (50085, 25) (50085,)



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```


```python
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
```

    (40068, 25) (40068,) (10017, 25) (10017,)


### Linear, DecisionTree, RandomForest, XGboost, SVM  Regression


```python
# LinearRegression Model

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("기본 점수 :",np.sqrt(mean_squared_error(y_test, y_pred)))

grid_params={
    'normalize':[True,False]
}

#model=modelfit(model, grid_params, X_train, y_train, X_test, y_test)
```

    기본 점수 : 4.079521574345239



```python
# DecisionTree Regression Model

from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("기본 점수 :",np.sqrt(mean_squared_error(y_test, y_pred)))

grid_params={
    'min_samples_leaf': [1, 2, 3],
    'max_depth': [1, 2, 3, 4, 5]
}

#model = modelfit(model, grid_params, X_train, y_train, X_test, y_test)
```

    기본 점수 : 5.37787443077946



```python
# RandomForest Regression Model
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("기본 점수 :",np.sqrt(mean_squared_error(y_test, y_pred)))

grid_params={
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [1,2,3,4],
}

#model = modelfit(model, grid_params, X_train, y_train, X_test, y_test)
```

    기본 점수 : 3.78798514057121



```python
# XGboost Regression Model
import xgboost as xgb 

model = xgb.XGBRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("기본 점수 :",np.sqrt(mean_squared_error(y_test, y_pred)))

grid_params={
    "colsample_bytree":[1.0],
    "min_child_weight":[1.0,1.2],
    'max_depth': [3,4,6,7,8,9,10], 
    'n_estimators': [250,500,1000,1500,2000]
}

#model = modelfit(model, grid_params, X_train, y_train, X_test, y_test)
```

    기본 점수 : 3.795951511323613



```python
# Support Vector Regression Model
from sklearn.svm import SVR

model=SVR()
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

#model = modelfit(model, grid_params, X_train, y_train, X_test, y_test)
```

    기본 점수 : 4.268740162818391

