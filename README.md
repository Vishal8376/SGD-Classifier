# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Vishal S
RegisterNumber: 212224040364  
*/
```
```
#ML 4


mport numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target']=dataset.target
df.head()
x=df.drop(columns=['AveOccup','target'])
y=df[['AveOccup','target']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
y_train=scaler_y.fit_transform(y_train)
x_test=scaler_x.transform(x_test)
y_test=scaler_y.transform(y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
model=MultiOutputRegressor(sgd)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("Predictions\n",y_pred[:5])
```
## Output:
![image](https://github.com/user-attachments/assets/74c1e3d9-27f1-4d6d-9df7-7f8ecd0b1b1c)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
