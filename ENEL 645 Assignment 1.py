#!/usr/bin/env python
# coding: utf-8

# ### By Michael Le (UCID: 10104883)

# In[1]:


import numpy as np  # Import NumPy
import pandas as pd # Import Pandas
import matplotlib.pyplot as plt # Matplotlib Data Visualization Library
import seaborn as sns # Seaborn Data Visualization Library

from sklearn.model_selection import train_test_split # Import Scikit-Learn train_test_split method
from sklearn.linear_model import LinearRegression # Import LinearRegression from sklearn
from sklearn.metrics import mean_squared_error # Import mean_squared_error method from Scikit-Learn metrics


# In[2]:


# Upload the CSV data and convert it to a Pandas DataFrame, where Community Name is the Index column
df = pd.read_csv('./reduced_version_data_ENEL_645.csv', index_col="Community Name")
df.head()


# In[3]:


# View dimensions of the DataFrame
df.shape


# In[4]:


# Check for null values
df.isnull().sum() 


# In[5]:


# Use get_dummies to convert categorical variables into separate indicator columns of 0s and 1s and create a new DataFrame 
# called df_dummies

df_dummies = pd.get_dummies(df, columns=['Sector', 'Group Category', 'Category', 'Month'])
df_dummies


# In[6]:


# View columns in df_dummies to ensure no columns are missing
list(df_dummies.columns)


# In[7]:


# Create Features Matrix X
X = df_dummies[df_dummies.columns[1:]]
# Create output vector y, which is the Crime Count column in df_dummies
y = df_dummies[['Crime Count']]
# Split arrays or matrices into random train and test subsets. Use 70/30 split.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=956)


# In[8]:


pd.set_option('display.max_columns', None) # Set pandas option to view all columns
X.head() # View preview of Features Matrix X


# In[9]:


# View dimensions of the Features Matrix X and Output Column y for both training and testing sets
print(f"Dimensions of X_train: {X_train.shape}")
print(f"Dimensions of X_test: {X_test.shape}")
print(f"Dimensions of y_train: {y_train.shape}")
print(f"Dimensions of y_test: {y_test.shape}")


# ### Algorithm Description
# In the code block below, I am instantiating a Linear Regression model. I have decided to keep all features (except for Community Name which is now the index) and apply one hot encoding with the get_dummies() method from pandas for feature engineering, in order to convert categorical data variables like Sector, Group Category, Category, and Month so they can be used in my model to improve predictions. I fit the Linear Regression model to the training data, which is 70% of the total data because this was arbitrarily chosen in the assignment description. Once the model is trained, I use the newly-trained model to predict Crime Count values of the remaining 30% of the total data (the testing data).
# 
# Once the prediction is calculated, I print the coefficients of the model. I also calculate and print the mean-squared error using the true crime count values in the testing set, compared to the predicted values.

# In[10]:


# Instantiate Linear Regression model
model = LinearRegression()
# Train the Linear Regression model using training data
model.fit(X_train, y_train)
# Use trained model to predict output values if the test features matrix is used
y_test_pred = model.predict(X_test)

# The coefficients of the model
print("Coefficients: \n", model.coef_ , "\n")

# Use the true labels and predicted labels for the testing set to determine mean squared error
testing_mse = mean_squared_error(y_test, y_test_pred) 
print(f"Mean Squared Error (Testing Data): {testing_mse}")


# ### Analysis
# 
# The performance of my model is evaluated based on mean-squared-error cost function, which is equal to 453.71635510105625. This is expected because we are using a real-world dataset from Open Calgary. The data in a Community Crime and Disorder Statistics dataset is likely complex enough that applying a linear regression model to it will not accurately capture and predict the number of crimes in each community center.

# ### Screenshot: Results
# ![image.png](attachment:image.png)

# In[11]:


# Plot Test Output Vector and Predictions
plt.figure(figsize=(8, 4), dpi=200)
with sns.axes_style('darkgrid'):
  ax = sns.regplot(x=y_test, y=y_test_pred, color="salmon")
  ax.set(title="Plot of Calgary Community Crime and Disorder Statistics Dataset and Linear Regression Model Fit") 
plt.xlabel('Crime Count')
plt.ylabel('Crime Count');

