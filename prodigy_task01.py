import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

df=pd.read_csv('house_price_predict.csv')

df.head(5)

df['sqft_living'].info()

df.head()

df['sqft_living'] = df['sqft_living'].astype(float)

df['sqft_lot'] = df['sqft_lot'].astype(float)

df['sqft_above'] = df['sqft_above'].astype(float)

df['sqft_basement'] = df['sqft_basement'].astype(float)

df['sqr_ft'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

df['sqr_ft'].info()

df = df.drop(columns=['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement'])


df.head()


df['country'].unique()


df.isnull().sum()

X = df[['sqr_ft', 'bedrooms', 'bathrooms']]
y = df['price']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


X_train.head()

X_train.info()

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score

# applying cross val score

pt = PowerTransformer()
X_transformed2 = pt.fit_transform(X)

lr = LinearRegression()
np.mean(cross_val_score(lr,X_transformed2,y,scoring='r2'))

