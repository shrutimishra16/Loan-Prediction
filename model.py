from sklearn import linear_model
lm = linear_model.LinearRegression()

import pandas as pd
import pickle

df = pd.read_csv('loan_predict_train.csv')

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

y = df ['LoanAmount'] # dependent variable
x = df [['Education','ApplicantIncome']] # independent variable

lm.fit(x,y) # fitting the model
pickle.dump(lm ,open('model.pkl','wb')) # save the model
print(lm.predict([[1,3000]])) # format of input
print(f'score : {lm.score(x,y)}')