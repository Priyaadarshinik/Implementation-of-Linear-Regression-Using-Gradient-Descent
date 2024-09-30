# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import the required library and read the dataframe.
3. Set variables for assigning dataset values.
4. Write a function computeCost to generate the cost function.
5. Perform iterations og gradient steps with learning rate.
6. Plot the Cost function using Gradient Descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PRIYAADARSHINI.K
RegisterNumber:  212223240126
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
dataset=pd.read_csv('50_Startups.csv')
print("dataset.tail()")
print(dataset.head())

print("dataset.tail()")
print(dataset.tail())
print(dataset.info())
x=(dataset.iloc[1:, :-2].values)
x1=x.astype(float)
y=(dataset.iloc[1:,-1].values).reshape(-1,1)
print(x)
print(y)
scaler=StandardScaler()
x1_scaled=scaler.fit_transform(x)
y1_scaled=scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)

def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
print(new_Scaled)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted valeue: {pre}")
```
## Output:

![image](https://github.com/user-attachments/assets/e965736d-10d2-4c4e-85cd-054d04b43fce)

![image](https://github.com/user-attachments/assets/0dc68bc5-fe44-4ab4-91d5-ee67793fd070)

![image](https://github.com/user-attachments/assets/51432fad-c2ea-4961-875e-60ca2ddc7f8e)
![image](https://github.com/user-attachments/assets/5faa42c0-4640-40bc-a358-6ed3f8909607)

![image](https://github.com/user-attachments/assets/83ce172c-a843-45e7-92a7-73d4153a474c)
![image](https://github.com/user-attachments/assets/c75e6ed2-f284-4a47-a248-1b9ee37d9043)

![image](https://github.com/user-attachments/assets/c962a08d-4bc0-4684-a524-fb3487e9a471)
![image](https://github.com/user-attachments/assets/42d41bc2-1151-4171-810a-a07915a35221)

![image](https://github.com/user-attachments/assets/fd7e027f-1d94-4a3f-9c25-53112cdd276e)
![image](https://github.com/user-attachments/assets/79acdd53-5267-48be-8822-251f1675e53a)

![image](https://github.com/user-attachments/assets/a52078fe-567c-44f5-a0b4-55a60483d5d8)

![image](https://github.com/user-attachments/assets/9c149ecd-36c1-464f-af07-918940521790)

![image](https://github.com/user-attachments/assets/0bec6f12-334a-49f3-8129-a81464116d1f)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
