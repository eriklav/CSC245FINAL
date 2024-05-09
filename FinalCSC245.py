#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Load the dataset
file_path = 'House_Rent_Dataset.csv'
df = pd.read_csv(file_path, encoding='ascii')

# Display the first few rows of the dataframe
df.head()


# In[4]:


# Check for missing values and data types
df.info()

# Display unique values for categorical columns to understand the encoding needs
categorical_columns = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
for col in categorical_columns:
    print(f'Unique values in {col}:', df[col].unique())


# In[8]:


# One-hot encode categorical variables
categorical_columns = ['Floor', 'Area Type', 'Area Locality', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
df_encoded = pd.get_dummies(df, columns=categorical_columns)


# In[9]:


# Split the data into features and target variable
X = df_encoded.drop('Rent', axis=1)  # Features
y = df_encoded['Rent']  # Target variable


# In[10]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display the RMSE
print('Root Mean Squared Error:', rmse)


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Selecting the feature and target variable
X = df[['Bathroom']]  # Feature: Number of Bathrooms
y = df['Size']  # Target: Size of the property

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest Regressor
model_size = RandomForestRegressor(n_estimators=100, random_state=42)
model_size.fit(X_train, y_train)

# Predicting on the test set
y_pred_size = model_size.predict(X_test)

# Calculating the Mean Squared Error and Root Mean Squared Error
mse_size = mean_squared_error(y_test, y_pred_size)
rmse_size = np.sqrt(mse_size)

# Display the RMSE
print('Root Mean Squared Error for Size prediction:', rmse_size)


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Set 'Bathroom' as the target variable
y = df['Bathroom']

# Use 'Rent' as the primary feature
X = df[['Rent']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display the RMSE
print('Root Mean Squared Error:', rmse)


# In[ ]:




