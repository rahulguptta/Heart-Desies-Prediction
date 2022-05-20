# Heart Disease Prediction by manually created Logistic Regression 


## Introduction

- This is project about creating self Logistic regression classifier using python and Comparing the performance with the model created using the inbuilt library scikit learn.

- The dataset for this project is "Heart Failure Prediction Dataset" downloaded from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

- The goal project is creating a classifier using logistic regression to predict if a has hear disease. 

- The outflow of the project is as:

    a. Overview of the data 

    b. Visualization

    c. Data Processing

    d. Classifier Modeling

    e. Classifier using Scikit learn

    f. Comparison and conclusion

## Overview of the data 

```python
# Imporiting the required libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('dark_background')
import warnings
warnings.filterwarnings('ignore')

# Importing the data
df = pd.read_csv('heart.csv')

#The simple overview of the data is
df.head(5)
```
<center>
<div  style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPainType</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>RestingECG</th>
      <th>MaxHR</th>
      <th>ExerciseAngina</th>
      <th>Oldpeak</th>
      <th>ST_Slope</th>
      <th>HeartDisease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>289</td>
      <td>0</td>
      <td>Normal</td>
      <td>172</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>F</td>
      <td>NAP</td>
      <td>160</td>
      <td>180</td>
      <td>0</td>
      <td>Normal</td>
      <td>156</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>283</td>
      <td>0</td>
      <td>ST</td>
      <td>98</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>F</td>
      <td>ASY</td>
      <td>138</td>
      <td>214</td>
      <td>0</td>
      <td>Normal</td>
      <td>108</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>M</td>
      <td>NAP</td>
      <td>150</td>
      <td>195</td>
      <td>0</td>
      <td>Normal</td>
      <td>122</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</center>

```python
#Getting a genera information about the features
df.info() 
```
<div class="output stream stdout">
<pre><code>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 918 entries, 0 to 917
Data columns (total 12 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Age             918 non-null    int64  
 1   Sex             918 non-null    object 
 2   ChestPainType   918 non-null    object 
 3   RestingBP       918 non-null    int64  
 4   Cholesterol     918 non-null    int64  
 5   FastingBS       918 non-null    int64  
 6   RestingECG      918 non-null    object 
 7   MaxHR           918 non-null    int64  
 8   ExerciseAngina  918 non-null    object 
 9   Oldpeak         918 non-null    float64
 10  ST_Slope        918 non-null    object 
 11  HeartDisease    918 non-null    int64  
dtypes: float64(1), int64(6), object(5)
memory usage: 86.2+ KB
</code></pre>
</div>
**Observations**

1. 918 samples with zero null 

2. 11 features where 

    **float** - ('Oldpeak')

    **int** - ('Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR')

    **Object** -  ('Sex', 'ChestPainType', 'RestingECG', 'ExerciseAgina')

Ok, so we have our features and their type. As there are no null value, we will check if Object features have any typo. We will also check if the float or float type features have any nan values. 

```python
#Categorical
features_cat = df.select_dtypes(include = ['object'])
for col in features_cat.columns:
  print(col, df[col].unique())

#nan
print('\n', 'Number of nans', df.isna().sum().sum())
```
<div class="output stream stdout">
<pre><code>Sex [&#39;M&#39; &#39;F&#39;]
ChestPainType [&#39;ATA&#39; &#39;NAP&#39; &#39;ASY&#39; &#39;TA&#39;]
RestingECG [&#39;Normal&#39; &#39;ST&#39; &#39;LVH&#39;]
ExerciseAngina [&#39;N&#39; &#39;Y&#39;]
ST_Slope [&#39;Up&#39; &#39;Flat&#39; &#39;Down&#39;]
 Number of nans 0
</code></pre>
</div>
**Observations: ** Therefore, no typos and no nan values are present.

Now we will see how many categorical variable are present in our dataset.
```python
#Checking if categorical
feature_num = df.select_dtypes(include = [np.number])
for col in feature_num.columns:
  print(col, len(df[col].unique()))
```
<div class="output stream stdout">
<pre><code>Age 50
RestingBP 67
Cholesterol 222
FastingBS 2
MaxHR 119
Oldpeak 53
HeartDisease 2
</code></pre>
</div>


Looks like 'FastingBS' is also a categorical variable. Therefore, there are total 6 categorical variables (1 int and 5 objects.)

## Visualization

General view of the target variable with respect to two features look like

```python
xydf = df[['Age', 'Cholesterol', 'HeartDisease']]
xy0 = xydf[xydf['HeartDisease'] == 0]
xy1 = xydf[xydf['HeartDisease'] == 1]
fig = plt.gcf()
fig.set_size_inches(8,6)
sns.scatterplot(x = 'Age', y = 'Cholesterol', data = xy1, marker = 'o', color = 'green', label = '1');
sns.scatterplot(x = 'Age', y = 'Cholesterol', data = xy0, marker = 'o', color = 'red', label = '0');
plt.show()
```
![](/images/a9fb547c24449e23d204ff8e85c092b9d0c712af.png)
------



Let us see now if there is any correlation among the features.

```python
#Visualizing the relation among numerical features
sns.heatmap(df.corr(), annot = True)
```

![](/images/70b6403286511fd879cd7dccd5126ca1efd7c328.png)

**Observations**

1. There very small co-relation among features like MaxHR slightly decreases with age.
2. Heart Disease mostly depends on Age, Cholesterol, FastingBP, and MaxHR.
3. Basically the features are not highly co-related.

There are many things can be explored with the visualizations only. Some observations are listed below:

1. Is heart disease gender biased?

```python
sns.swarmplot(x='Sex', y='Age', hue = 'HeartDisease', data = df);
plt.show()
```

 																 ![](images\c9ebf8697133eb935c690adf045304ac8f590073.png)

**Observations**

- The first thing is the data has more samples of Male than female. 

- Look like Male have more heart problem than female and the tendency of having heart disease increases with age.

- Female mostly get heart disease after 50 but the chance of getting heart disease in male starts even after 40.

2. Do men have more cholesterol than women?

```python
sns.swarmplot(x='Sex', y='Cholesterol', hue = 'HeartDisease', data = df);
plt.show();

fig = plt.gcf();
fig.set_size_inches(15, 6);
sns.swarmplot(x='Age', y='Cholesterol', hue = 'Sex', data = df);
plt.show();
```

![](/images/f6a49c8a275bf7979a99e1457f3e76dbcafd36f2.png)

### Distribution of Continuous Variables

```python
# Distribution of Continuous Variables
plt.style.use('dark_background')
for col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
  sns.kdeplot(df[col]);
  plt.show()
```

![](/images/2a48297ba3aaccde6c61493cc8730d5c7337810d.png)

![](/images/5ff16eee992b6e7c1fed52c1b75d020a4f9b09bb.png)

![](/images/6becf59c94e5d8663aca6be06d7f5f07255dbcea.png)

![](/images/a2ea88a89a3b12c6e0f9c35acd6de3745c28d86d.png)

![](/images/aa407a749f57c648aa575409c94316959d198294.png)

**Observations**

- None of the continuous variables are highly skewed. Therefore, no log transformation will be required.

## Data Processing

First of all we will split the features from target variable. The features which have categorical values will be encoded such that the features will be turned into binary features that are “one-hot” encoded, meaning that if a feature is represented by that column, it receives a 1 Otherwise, it receives a 0.

```python
# Separating the features and Target
target = df['HeartDisease']
features = df.drop('HeartDisease', axis = 1)

# Encoding the categorical variables

def LabelEncoder(df, col):
  unique_values = list(df[col].unique())
  for value in unique_values:
    column = col + '_' + value
    df[column] = df[col].apply(lambda x : 1 if x == value else 0)
  df.drop(col, axis  = 1, inplace = True)
  return(df)

for col in ['ChestPainType', 'RestingECG', 'ST_Slope', 'Sex']:
  features = LabelEncoder(features, col)

#Encoding ExercisAngina
features['ExerciseAngina'] = features['ExerciseAngina'].apply(lambda x : 1 if x == 'Y' else 0)

#Trasnformation of continuous variables
def scalar(df, col):
  max = df[col].max()
  min = df[col].min()
  range = max - min
  df[col] = (df[col] - min)/range
  return(df)

for col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
  scalar(features, col)

features.head(5)
```
<center>
<div  style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>MaxHR</th>
      <th>ExerciseAngina</th>
      <th>Oldpeak</th>
      <th>ChestPainType_ATA</th>
      <th>ChestPainType_NAP</th>
      <th>ChestPainType_ASY</th>
      <th>ChestPainType_TA</th>
      <th>RestingECG_Normal</th>
      <th>RestingECG_ST</th>
      <th>RestingECG_LVH</th>
      <th>ST_Slope_Up</th>
      <th>ST_Slope_Flat</th>
      <th>ST_Slope_Down</th>
      <th>Sex_M</th>
      <th>Sex_F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.244898</td>
      <td>0.70</td>
      <td>0.479270</td>
      <td>0</td>
      <td>0.788732</td>
      <td>0</td>
      <td>0.295455</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.428571</td>
      <td>0.80</td>
      <td>0.298507</td>
      <td>0</td>
      <td>0.676056</td>
      <td>0</td>
      <td>0.409091</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.183673</td>
      <td>0.65</td>
      <td>0.469320</td>
      <td>0</td>
      <td>0.267606</td>
      <td>0</td>
      <td>0.295455</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.408163</td>
      <td>0.69</td>
      <td>0.354892</td>
      <td>0</td>
      <td>0.338028</td>
      <td>1</td>
      <td>0.465909</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.530612</td>
      <td>0.75</td>
      <td>0.323383</td>
      <td>0</td>
      <td>0.436620</td>
      <td>0</td>
      <td>0.295455</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</center>

- We can see that the feature "ChestPainType" is turned into 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_ASY', and 'ChestPainType_TA'. Similarly the other categorical features are also converted.
- Therefore, now every categorical feature is encoded with 0 and 1. 
- Every Numerical feature is scaled using between 0 and 1
- Now the dataset is ready for preparing the model.

## Classifier Modeling

### Converting the data into matrix form

It is known that the hypothesis function for Logistic regression classifier is
![](/images/math1.png)

```python
n = features.shape[0] #number of samples
m = features.shape[1] #number of features
features_mat = features.to_numpy() 
X = np.hstack((np.ones((n, 1)), features_mat)) #adding column of 1
Y = target.to_numpy()

print("Sample of Features matrix: ", "\n\n", X[0:3, :], '\n\n', 'Size = ', (n,m)) 
print("\n", "Sample Target vector: ", "\n\n", Y[:10],  '\n\n', 'Size = ', (n))
```

<div class="output stream stdout">
<pre><code>Sample of Features matrix:  

 [[1.         0.24489796 0.7        0.47927032 0.         0.78873239
  0.         0.29545455 1.         0.         0.         0.
  1.         0.         0.         1.         0.         0.
  1.         0.        ]
 [1.         0.42857143 0.8        0.29850746 0.         0.67605634
  0.         0.40909091 0.         1.         0.         0.
  1.         0.         0.         0.         1.         0.
  0.         1.        ]
 [1.         0.18367347 0.65       0.46932007 0.         0.26760563
  0.         0.29545455 1.         0.         0.         0.
  0.         1.         0.         1.         0.         0.
  1.         0.        ]] 

 Size =  (918, 19)

 Sample Target vector:  

 [0 1 0 1 0 0 0 0 1 0] 

 Size =  918 
 </code></pre></div>

### Splitting into test and train

```python
ntrain = round(0.8*n) #test_size = 0.2
xtrain = X[0:ntrain, :]  
xtest = X[ntrain:n+1,:]

ytrain = Y[0:ntrain] 
ytest = Y[ntrain:]
```

### Functions involved in Logistic regression
![](/images/math2.png)

```python
# Hypothesis function
def sigma(xtrain,theta):
  z = np.matmul(xtrain, theta)
  ze = np.exp(-z)
  hx = 1 / (1 + ze)
  return(z, hx)

#Cost function 
def costf(z,hx,ytrain):
  c1 = np.matmul(np.log(hx), ytrain) #Y^T H(X)
  lhx = np.log(1 - hx)
  c2 = np.matmul(lhx, (1-ytrain))  #(1-Y)^T (log(1+exp(Z)))
  return(-c2-c1)
  

#Optimization of cost function
def Optimize(xtrain,ytrain,eta):
  n = np.shape(xtrain)[0]
  theta = np.random.random(np.shape(xtrain)[1]) #initialization of theta
  (z, hx) = sigma(xtrain,theta)
  cost_list = [costf(z,hx,ytrain)]  #list of cost
  cost_diff = cost_list[-1] 
  while(cost_diff > 0):
    nabla = np.matmul(xtrain.transpose(), (hx-ytrain))/n #the gredient
    theta = theta - eta*nabla
    (z, hx) = sigma(xtrain,theta)
    cost = costf(z,hx,ytrain)
    cost_list.append(cost)
    cost_diff = round(abs(cost_list[-2] - cost_list[-1]), 4)
  return(theta, cost_list)                                                     

#predicting labels
def predict(xtest,theta):
  (z, hx) = sigma(xtest,theta)
  ypred = np.array([1 if x>=0.5 else 0 for x in hx])
  return(ypred)

#Creatig confusion matrix
def creat_cm(ytest,ypred):
  cm = pd.DataFrame(columns=['predicted 0', 'predicted 1'])
  (zero, zeroOne, one, oneZero) = (0,0,0,0)
  for i in range(np.size(ytest)):
    if ytest[i] == 0 and ypred[i] == 0:
      zero = zero + 1
    if ytest[i] == 0 and ypred[i] == 1:
      zeroOne = zeroOne + 1
    if ytest[i] == 1 and ypred[i] == 1:
      one = one + 1
    if ytest[i] == 1 and ypred[i] == 0:
      oneZero = oneZero + 1
  
  cm.loc['Actual 0'] = [zero, zeroOne]
  cm.loc['Actual 1'] = [oneZero, one]
  return(cm)

#the Decision metrics
def metrics(cm):
  metric = {}
  n = cm['predicted 0'].sum() + cm['predicted 1'].sum()
  metric['accuracy'] = (cm['predicted 0'].loc['Actual 0'] + cm['predicted 1'].loc['Actual 1'])/n
  metric['precision'] = (cm['predicted 1'].loc['Actual 1'])/(cm['predicted 1'].loc['Actual 1'] + cm['predicted 1'].loc['Actual 0'])
  metric['recall'] = (cm['predicted 1'].loc['Actual 1'])/(cm['predicted 1'].loc['Actual 1'] + cm['predicted 0'].loc['Actual 1'])
  metric['F-1 Score'] = 2*metric['precision']*metric['recall']/(metric['precision']+metric['recall'])
  return(metric)

#classifier Modeling 
def classifier(xtrain,ytrain,eta_list,xtest,ytest):
  reports = pd.DataFrame(columns = ['eta', 'accuracy', 'precision', 'recall', 'F-1 Score'])
  predicted_values = pd.DataFrame()
  predicted_values['Actual label'] = ytest
  cost_variation = {}
  theta_variation = {}
  for eta in eta_list:
    (theta, cost_list)  = Optimize(xtrain,ytrain,eta)
    cost_variation[eta] = cost_list
    theta_variation[eta] = theta
    ypred = predict(xtest,theta)
    predicted_values['label_'+str(eta)] = ypred
    cm = creat_cm(ytest,ypred)
    metric = metrics(cm)
    metric['eta'] = eta
    reports = reports.append(metric, ignore_index=True)
  
  return(cost_variation, reports, predicted_values,theta_variation)
```

### Classification

```python
#classification results
eta_list = [0.0001, 0.001, 0.01, 0.1]
(cost_variation, reports, predicted_values,theta_variation) = classifier(xtrain, ytrain, eta_list, xtest, ytest)

#Visualization of cost 
plt.style.use('grayscale')
fig = plt.gcf();
fig.set_size_inches(8,6)
x = list(range(len(y)))
sns.lineplot(range(len(cost_variation[0.0001])),cost_variation[0.0001],label=0.0001, c='r')
sns.lineplot(range(len(cost_variation[0.001])),cost_variation[0.001],label=0.001, c='g')
sns.lineplot(range(len(cost_variation[0.01])),cost_variation[0.01],label=0.01, c='b')
sns.lineplot(range(len(cost_variation[0.1])),cost_variation[0.1],label=0.1, c='y')
plt.title('Variation of cost for different learning rate')
plt.xlim(0,30000)
plt.legend()
plt.show()

#Performance of classifier
print(reports)
```

![](/images/233e1b2ff6f56576e2b6567b2e2810a90ace0344.png)

<div class="output stream stdout">
<pre><code>      eta  accuracy  precision    recall  F-1 Score
0  0.0001  0.749091   0.734375  0.728682   0.731518
1  0.0010  0.763636   0.775862  0.697674   0.734694
2  0.0100  0.774545   0.801802  0.689922   0.741667
3  0.1000  0.774545   0.801802  0.689922   0.741667
</code></pre>
</div>
**Observations**

- The optimization in this case is best for learning rate (eta) = 0.1
- The accuracy of the classifier is approximately same for all the learning rates.
- As this the project on the prediction of Heart disease, we would like the maximum recall (i.e., maximum of the sample should be detected positive if it is actually positive)
- The maximum recall can be obtained using eta = 0.0001.

### Prediction

```python
#The confusion matrix for eta = 0.0001
Confusion_matrix = creat_cm(ytest, predicted_values['label_0.0001'])
print('Confusion Matix', '\n', Confusion_matrix, '\n')

#Visulaizing the prediction
df_pre = df.iloc[ntrain:]
df_pre = df_pre[['Age', 'Cholesterol']]
df_pre['Actual label'] = list(predicted_values['Actual label'])
df_pre['predicted'] = list(predicted_values['label_0.0001'])

true = df_pre[df_pre['Actual label'] == df_pre['predicted']]
false = df_pre[df_pre['Actual label'] != df_pre['predicted']]
sns.scatterplot(x = 'Age', y = 'Cholesterol', data = true, marker = '*', color = 'blue', label = 'predicted_true');
sns.scatterplot(x = 'Age', y = 'Cholesterol', data = false, marker = '+', color = 'red', label = 'predicted_false');
plt.title('Visualization of prediction')
plt.show()
```
<div class="output stream stdout">
<pre><code>Confusion Matix 
          predicted 0 predicted 1
Actual 0         112          34
Actual 1          35          94 
</code></pre>
</div>

![](/images/21dfcae8692c332f8c1899f0c8dc27e750b0338c.png)

## Classifier Modeling using Sklearn

```python
#Importing the dataset
df = pd.read_csv('heart.csv')

#Separating Features and Target
target = df['HeartDisease']
features = df.drop('HeartDisease', axis = 1)

# Encoding the categoriacal variables
features = pd.get_dummies(features)

# Scaling the numeric features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(features)

# Splitting the train and test data 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size = 0.3, random_state = 0)

#Classifier Modeling
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)

#Predictions
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
cm = confusion_matrix(ytest, ypred)
report = classification_report(ytest, ypred)
accuracy = accuracy_score(ytest, ypred)
print('Acuracy = ', accuracy, '\n')
print(report)
```
<div class="output stream stdout">
<pre><code>Acuracy =  0.8369565217391305 

              precision    recall  f1-score   support
    
           0       0.82      0.77      0.79       113
           1       0.85      0.88      0.86       163
    
    accuracy                           0.84       276
   macro avg       0.83      0.83      0.83       276
weighted avg       0.84      0.84      0.84       276

</code></pre>
</div>

## Comparison and Conclusion

- The accuracy of the model created manually is 0.8 (eta = 0.0001) and the accuracy of the model created using Sklearn is 08.
- The recall of the model created manually is 0.8 (eta = 0.0001) and the recall of the model created using Sklearn is 08.
- The Sklearn Model performs slightly better than the model created manualy.
