#!/usr/bin/env python
# coding: utf-8

# 

# NAME: KOLETI SANKEERTHANA
# 
# AICTE ID: STU6820e2f2b36221746985714
# 
# PROJECT TITLE: EMPLOYEE SALARY PREDICTION
# 

# In[25]:


import pandas as pd


# In[26]:


data=pd.read_csv("C:\\Users\\saipa\\Downloads\\Telegram Desktop\\adult 3.csv")
data


# In[27]:


print(data.info())        # basic info


# In[28]:


print(data.head())        # first 5 rows


# In[29]:


print(data.tail())        # last 5 rows


# In[30]:


print(data.describe())    # statistics


# In[31]:


print(data.columns.tolist())


# In[32]:


X = data[['age','educational-num','hours-per-week','capital-gain','capital-loss']]
y= data['income']


# In[33]:


X = pd.get_dummies(data.drop('income', axis=1))
y = data['income']


# In[34]:


print(data.columns.tolist())
print(data.dtypes)


# In[35]:


print(y.value_counts())


# In[36]:


print(data.isnull().sum())# missing values count


# In[37]:


print(data.shape)         # rows, columns


# In[38]:


print(data.workclass.value_counts())


# In[39]:


data.workclass.replace({'?':'Others'},inplace=True)
print(data['workclass'].value_counts())


# In[40]:


print(data['occupation'].value_counts())


# In[41]:


data.occupation.replace({'?':'Others'},inplace=True)
print(data['occupation'].value_counts())


# In[42]:


data=data[data['workclass']!='Without-pay']
data=data[data['workclass']!='Never-worked']
print(data['workclass'].value_counts())


# In[43]:


print(data.relationship.value_counts())


# In[44]:


print(data.gender.value_counts())


# VISUALIZING THE DATA

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your dataset (use your own path if needed)
# df_adult = pd.read_csv("your_file.csv")  # Uncomment if needed

# Convert income to binary class: <=50K → 0, >50K → 1
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Extract the features for plotting
x_feature_1 = data['age'].values
x_feature_2 = data['hours-per-week'].values
y_target = data['income'].values

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Class 0 (<=50K)
ax.scatter(
    x_feature_1[y_target == 0],
    x_feature_2[y_target == 0],
    y_target[y_target == 0],
    color='blue',
    label='<=50K',
    alpha=0.5
)

# Plot Class 1 (>50K)
ax.scatter(
    x_feature_1[y_target == 1],
    x_feature_2[y_target == 1],
    y_target[y_target == 1],
    color='red',
    label='>50K',
    alpha=0.5
)

# Axis labels and title
ax.set_xlabel('Age')
ax.set_ylabel('Hours-per-week')
ax.set_zlabel('Income Class (0 or 1)')
ax.legend()
plt.title('3D Scatter Plot: Age vs Hours-per-week vs Income Class')
plt.tight_layout()
plt.show()


# ## Applying dimensionality reduction techniques for effective data visualization.

# In[65]:


from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
pca2.fit(X_train)

X_train_2D_pca = pca2.transform(X_train)

print(f'PCA Features: Number of samples and attributes: {X_train_2D_pca.shape}')


# In[66]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(projection='3d')

ax.scatter(X_train_2D_pca[y_train==0, 0], X_train_2D_pca[y_train==0, 1], y_train[y_train==0], color="b");
ax.scatter(X_train_2D_pca[y_train==1, 0], X_train_2D_pca[y_train==1, 1], y_train[y_train==1], color="r");

ax.set_xlabel("PCA-1")
ax.set_ylabel("PCA-2")
ax.set_zlabel("Target")

plt.tight_layout()
plt.show()


# In[46]:


#outlier detection
import matplotlib.pyplot as plt   #visualization
plt.boxplot(data['age'])
plt.show()


# In[47]:


data=data[(data['age']<=75)&(data['age']>=17)]


# In[48]:


plt.boxplot(data['age'])
plt.show()


# In[49]:


plt.boxplot(data['capital-gain'])
plt.show()


# In[50]:


plt.boxplot(data['capital-gain'])
plt.show()


# In[51]:


plt.boxplot(data['educational-num'])
plt.show()


# In[52]:


data=data[(data['educational-num']<=16)&(data['educational-num']>=5)]


# In[53]:


plt.boxplot(data['educational-num'])
plt.show()


# In[54]:


plt.boxplot(data['hours-per-week'])
plt.show()


# In[55]:


data=data.drop(columns=['education']) #redundant features removal


# In[56]:


data


# In[57]:


from sklearn.preprocessing import LabelEncoder   #import libarary
encoder=LabelEncoder()                       #create object
data['workclass']=encoder.fit_transform(data['workclass']) #7 categories   0,1, 2, 3, 4, 5, 6,
data['marital-status']=encoder.fit_transform(data['marital-status'])   #3 categories 0, 1, 2
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])      #5 categories  0, 1, 2, 3, 4
data['race']=encoder.fit_transform(data['race'])  
data['gender']=encoder.fit_transform(data['gender'])    #2 catogories     0, 1
data['native-country']=encoder.fit_transform(data['native-country'])


# In[58]:


data


# In[59]:


X=data.drop(columns=['income'])
y=data['income']
X


# ## Predicting all machine learning models score's by using pipeline model

# In[60]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


# In[67]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier())
])

pipeline.fit(X_train, y_train)

print("✅ Model trained with 5 features!")
print("✅ Test Accuracy:", pipeline.score(X_test, y_test))

# Save the model
joblib.dump(pipeline, 'best_model.pkl')
print("✅ best_model.pkl overwritten successfully!")


# In[71]:


import matplotlib.pyplot as plt
plt.bar(results.keys(), results.values(), color='orange')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(False)
plt.show()

