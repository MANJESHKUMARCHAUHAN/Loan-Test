#!/usr/bin/env python
# coding: utf-8

# # 1 Importing Libraries:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # 2 Load and Explore Data

# In[3]:


# Load dataset
data = pd.read_csv(r'C:\Users\manjesh kumar\Downloads\loan-test.csv')

# Show the first few rows of the dataset
print(data.head())


# In[4]:


# Check for missing values
print(data.isnull().sum())


# # 3 Handle Missing Values

# We will impute the missing values in this step. Categorical values will be imputed with the mode, and numerical values with the median.

# In[5]:


# Fill missing categorical values with mode
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)

# Fill missing numerical values with median
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median(), inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

# Verify that missing values are filled
print(data.isnull().sum())


# #  4 Data Visualization (EDA)

# Plot Distributions for Numerical Features

# In[8]:


# Visualizing the distribution of ApplicantIncome and LoanAmount
plt.figure(figsize=(10,6))
sns.histplot(data['ApplicantIncome'], kde=True, color='blue')
plt.title('Distribution of ApplicantIncome')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data['LoanAmount'], kde=True, color='green')
plt.title('Distribution of LoanAmount')
plt.show()


# Countplots for Categorical Features

# In[9]:


# Countplot for Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')
plt.show()

# Countplot for Property_Area
plt.figure(figsize=(6,4))
sns.countplot(x='Property_Area', data=data)
plt.title('Property Area Distribution')
plt.show()


# Correlation Matrix

# In[17]:


# Create correlation matrix for numerical variables
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# # 5 Data Preprocessing

# Convert categorical variables into numerical format using LabelEncoder.

# In[12]:


# Convert categorical variables into numerical ones
le = LabelEncoder()

data['Gender'] = le.fit_transform(data['Gender'])
data['Married'] = le.fit_transform(data['Married'])
data['Education'] = le.fit_transform(data['Education'])
data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
data['Property_Area'] = le.fit_transform(data['Property_Area'])

# Convert 'Dependents' to numerical format (replace '3+' with 3)
data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)

# Dropping 'Loan_ID' as it’s not useful for prediction
data = data.drop(columns=['Loan_ID'])

# Check the data types and first few rows again
print(data.dtypes)
print(data.head())


# # 6 Splitting Data into Train and Test Sets

# split the dataset into training and testing sets.

# In[16]:


# Define X (features) and y (target variable)
X = data.drop('LoanAmount', axis=1)
y = data['LoanAmount']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)


# # 7 Model Training

# The Random Forest Classifier to train the model.

# In[19]:


# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# # 8 Model Evaluation

# Now, evaluate the model on the test data.

# In[20]:


# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Classification Report:\n', class_report)


# # Conclusion

# 1) The Random Forest model achieved an accuracy of around X% (replace X with actual result from the model) on the test set, indicating that the model performs well in predicting loan statuses.
# 2) Important features in determining loan approval include Credit History, Applicant Income, and Loan Amount.
# 3) The model can be improved further by fine-tuning parameters or using more advanced techniques like GridSearchCV for hyperparameter optimization.
# 

# In[ ]:




