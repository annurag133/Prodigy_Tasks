#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 


# In[79]:


df=pd.read_csv('house_price_predict.csv')


# In[80]:


df.head(5)


# In[77]:


df['sqft_living'].info()


# In[81]:


df.head()


# In[82]:


df['sqft_living'] = df['sqft_living'].astype(float)


# In[83]:


df['sqft_lot'] = df['sqft_lot'].astype(float)


# In[84]:


df['sqft_above'] = df['sqft_above'].astype(float)


# In[85]:


df['sqft_basement'] = df['sqft_basement'].astype(float)


# In[86]:


df['sqr_ft'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']


# In[87]:


df['sqr_ft'].info()


# In[ ]:





# In[88]:


df = df.drop(columns=['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement'])


# In[ ]:





# In[89]:


df.head()


# In[90]:


df['country'].unique()


# In[91]:


df.isnull().sum()


# In[ ]:





# In[92]:


X = df[['sqr_ft', 'bedrooms', 'bathrooms']]
y = df['price']


# In[93]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[94]:


X_train.head()


# In[95]:


X_train.info()


# In[96]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[111]:


y_pred = model.predict(X_test)


# In[112]:


from sklearn.metrics import mean_squared_error, r2_score


# In[113]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[114]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[115]:


from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score


# In[116]:


# applying cross val score

pt = PowerTransformer()
X_transformed2 = pt.fit_transform(X)

lr = LinearRegression()
np.mean(cross_val_score(lr,X_transformed2,y,scoring='r2'))


# In[ ]:





# In[ ]:




