import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('./testing.csv')
x=df['tweet']
y=df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
#print(type(x_train))
x_train.to_csv("xtest.csv",index=False)