---
jupyter:
  colab:
    name: Classification Using Logistic Regression (Manual Scripting Vs
      Scikit-learn).ipynb
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="_cL1Czuv_Aqx"}
## Introduction
:::

::: {.cell .markdown id="51OrAvPrLmDX"}
1.  This is project about creating personal Logistic regression
    classifier and Comparing the performance with the model creation
    using the inbuilt library scikit learn.

2.  The dataset for this procject is \"Heart Failure Prediction
    Dataset\" downloaded from
    [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

3.  The outflow of the project is as:

a\. Overview of the data

b\. Visualization

c\. Data Processing

d\. Classifier Modeling

e\. Classifier using Scikit learn

f\. Comparision and coclustion
:::

::: {.cell .markdown id="Em53vNu4-_OU"}
\#Overview of the data
:::

::: {.cell .code execution_count="1" id="JNcX3DALRYCU"}
``` {.python}
# Imporiting the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('dark_background')
import warnings
warnings.filterwarnings('ignore')
```
:::

::: {.cell .code execution_count="2" id="ImJ4n_7SWeRQ"}
``` {.python}
# Importing the data
df = pd.read_csv('heart.csv')
```
:::

::: {.cell .code execution_count="3" colab="{\"height\":206,\"base_uri\":\"https://localhost:8080/\"}" id="JCGsIfrvzZAO" outputId="dd9c4838-b9b6-49ba-afea-d5cdb01cc6a1"}
``` {.python}
#The simple overview of the data is
df.head(5)
```

::: {.output .execute_result execution_count="3"}
```{=html}
  <div id="df-861a6f12-535a-4a01-a20c-5fecb306e86f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-861a6f12-535a-4a01-a20c-5fecb306e86f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-861a6f12-535a-4a01-a20c-5fecb306e86f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-861a6f12-535a-4a01-a20c-5fecb306e86f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="yijUQUpS1Wuw" outputId="1af1ad55-0f74-459b-a03c-e6ae246d3197"}
``` {.python}
#Finding a general sense of the data
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
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
:::
:::

::: {.cell .code execution_count="6" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="f4tCRD9Jsrc3" outputId="06d6fcaa-851f-4c6c-df5b-e9b3fc47c7ce"}
``` {.python}
#Categorical
features_cat = df.select_dtypes(include = ['object'])
for col in features_cat.columns:
  print(col, df[col].unique())
```

::: {.output .stream .stdout}
    Sex ['M' 'F']
    ChestPainType ['ATA' 'NAP' 'ASY' 'TA']
    RestingECG ['Normal' 'ST' 'LVH']
    ExerciseAngina ['N' 'Y']
    ST_Slope ['Up' 'Flat' 'Down']
:::
:::

::: {.cell .code execution_count="8" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="0hWqsrgYvyVr" outputId="0e02bbb9-9f32-4cb5-b18e-1ef9093af295"}
``` {.python}
#nan
df.isna().sum().sum()
```

::: {.output .execute_result execution_count="8"}
    0
:::
:::

::: {.cell .code execution_count="12" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="2VlgitlrxBli" outputId="a227283b-e39d-4154-96ab-7e5c026dbe74"}
``` {.python}
#Checking if categorical
feature_num = df.select_dtypes(include = [np.number])
for col in feature_num.columns:
  print(col, len(df[col].unique()))
```

::: {.output .stream .stdout}
    Age 50
    RestingBP 67
    Cholesterol 222
    FastingBS 2
    MaxHR 119
    Oldpeak 53
    HeartDisease 2
:::
:::

::: {.cell .markdown id="uX3pqSqz3Ot0"}
**Observations**

1.  918 samples with zero null
2.  11 features where

a\. 1 float - Oldpeak, from above information it looks like non
categorical

b\. 4 int - Age, RestingBP, Cholesterol, FastingBS, MaxHR all are non
categorical features

c\. Age, RestingBP, Cholesterol, FastingBS, MaxHR, and Oldpeak are
continuous variable

c\. Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope are the
categorical features

d\. HeartDisease is the target variable with 0 and 1.

1.  Looks like there is no typo in the dataset
:::

::: {.cell .markdown id="-rmQVKBcMYM7"}
# Visualization
:::

::: {.cell .code colab="{\"height\":279,\"base_uri\":\"https://localhost:8080/\"}" id="dpIjRrdl8kbN" outputId="dc551490-09a8-4950-a62f-81e0d716afa4"}
``` {.python}
#Genrel view of target variable the
xydf = df[['Age', 'Cholesterol', 'HeartDisease']]
xy0 = xydf[xydf['HeartDisease'] == 0]
xy1 = xydf[xydf['HeartDisease'] == 1]
sns.scatterplot(x = 'Age', y = 'Cholesterol', data = xy1, marker = '*', color = 'blue', label = '1');
sns.scatterplot(x = 'Age', y = 'Cholesterol', data = xy0, marker = '+', color = 'yellow', label = '0');
plt.show()
```

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/e25ab2f2e1173aeff18b1205980e8ca00ccdda72.png)
:::
:::

::: {.cell .code colab="{\"height\":344,\"base_uri\":\"https://localhost:8080/\"}" id="wzgU9IjUjTll" outputId="b879f8f1-bafd-42ca-9121-6066a3503d6a"}
``` {.python}
#Visualizing the relation among numerical features
sns.heatmap(df.corr(), annot = True)
```

::: {.output .execute_result execution_count="45"}
    <matplotlib.axes._subplots.AxesSubplot at 0x7ff5d0260590>
:::

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/d4a988e6280b20e89a71bf087502abda12d6564e.png)
:::
:::

::: {.cell .markdown id="FRV6fw2Ak2FZ"}
1.  There very small co-relation among features like MaxHR slightly
    decreases with age.
2.  Heart Disease mostly depends on Age, Cholesterol, FastingBP, and
    MaxHR.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="RIzV0bwRdym7" outputId="6fb74dc4-160d-410f-efc9-2de4a54cceee"}
``` {.python}
#Visualizing Categoriacal features
categories = df.select_dtypes(include='object')
categories.columns
```

::: {.output .execute_result execution_count="5"}
    Index(['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], dtype='object')
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="-bAUkMDie1Yc" outputId="4cd9e565-9e08-4b0b-acbb-3ecdebc8548c"}
``` {.python}
for col in categories.columns:
  print(col, categories[col].value_counts(), '\n')
```

::: {.output .stream .stdout}
    Sex M    725
    F    193
    Name: Sex, dtype: int64 

    ChestPainType ASY    496
    NAP    203
    ATA    173
    TA      46
    Name: ChestPainType, dtype: int64 

    RestingECG Normal    552
    LVH       188
    ST        178
    Name: RestingECG, dtype: int64 

    ExerciseAngina N    547
    Y    371
    Name: ExerciseAngina, dtype: int64 

    ST_Slope Flat    460
    Up      395
    Down     63
    Name: ST_Slope, dtype: int64 
:::
:::

::: {.cell .markdown id="qh806f_KgCEa"}
1.  There are more samples corresponding to Male compared to Females.
2.  547 people out of 918 (arround 60 % people) do not do exercise.
:::

::: {.cell .code colab="{\"height\":279,\"base_uri\":\"https://localhost:8080/\"}" id="LK0HCnjcf8jN" outputId="51510fce-f14f-46e1-f0c3-753a018fb490"}
``` {.python}
# Let us see if there is any relation of sex, and age with Heart disease
sns.swarmplot(x='Sex', y='Age', hue = 'HeartDisease', data = df);
plt.show()
```

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/c9ebf8697133eb935c690adf045304ac8f590073.png)
:::
:::

::: {.cell .markdown id="_rJa6WXBhjgV"}
1.  Look like Male have more heart problem than female and the tendency
    of having heart disease increases with age.

2.  Female mostly get heart disease after 50 but the chance of getting
    heart disease in male starts even after 40.
:::

::: {.cell .code colab="{\"height\":279,\"base_uri\":\"https://localhost:8080/\"}" id="pIJxjSqihjMc" outputId="8915ec27-cb25-4190-8c1c-9b08be9bb1fb"}
``` {.python}
# Let us see if there is any relation of sex, and age with Heart disease
sns.swarmplot(x='Sex', y='Cholesterol', hue = 'HeartDisease', data = df);
plt.show();
```

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/f6a49c8a275bf7979a99e1457f3e76dbcafd36f2.png)
:::
:::

::: {.cell .code colab="{\"height\":388,\"base_uri\":\"https://localhost:8080/\"}" id="C5aAm0x1kSfu" outputId="828a80cd-8be8-497f-a574-8ba80d14aad1"}
``` {.python}
fig = plt.gcf();
fig.set_size_inches(15, 6);
sns.swarmplot(x='Age', y='Cholesterol', hue = 'Sex', data = df);
plt.show();
```

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/15c5ebe2a1be3292ca286b05e2e7e59ce962938a.png)
:::
:::

::: {.cell .markdown id="iTyKPrtTjnUF"}
1.  Male and Female both have arround same amount of Cholesterol which
    between 200 and 300.
2.  Male and Female both have arround same amount of Cholesterol
    irrespective of the age after 40.
:::

::: {.cell .code colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="qg3Mf-otjl1c" outputId="b497e632-8c58-4f30-e821-eb1cc755374c"}
``` {.python}
# Distribution of Continuous Variables
for col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
  sns.kdeplot(df[col]);
  plt.show()
  print('\n')
```

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/5ff16eee992b6e7c1fed52c1b75d020a4f9b09bb.png)
:::

::: {.output .stream .stdout}
:::

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/2a48297ba3aaccde6c61493cc8730d5c7337810d.png)
:::

::: {.output .stream .stdout}
:::

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/aa407a749f57c648aa575409c94316959d198294.png)
:::

::: {.output .stream .stdout}
:::

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/a2ea88a89a3b12c6e0f9c35acd6de3745c28d86d.png)
:::

::: {.output .stream .stdout}
:::

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/6becf59c94e5d8663aca6be06d7f5f07255dbcea.png)
:::

::: {.output .stream .stdout}
:::
:::

::: {.cell .markdown id="R7rJFGEbcFdu"}
None of the continuous variables are highly skewed. Therefore, no log
transformation will be required.
:::

::: {.cell .markdown id="2T9EXyHbcuIQ"}
# Data Processing
:::

::: {.cell .code colab="{\"height\":206,\"base_uri\":\"https://localhost:8080/\"}" id="FQsAvsGOcm1D" outputId="f2b98a13-4b2f-461e-9b81-0c362b82aa81"}
``` {.python}
# Separating the features and Target
target = df['HeartDisease']
features = df.drop('HeartDisease', axis = 1)
features.head(5)
```

::: {.output .execute_result execution_count="8"}
```{=html}
  <div id="df-f27deda0-33f0-4732-ae3f-ad9e901b60b5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f27deda0-33f0-4732-ae3f-ad9e901b60b5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f27deda0-33f0-4732-ae3f-ad9e901b60b5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f27deda0-33f0-4732-ae3f-ad9e901b60b5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code colab="{\"height\":270,\"base_uri\":\"https://localhost:8080/\"}" id="vrHMF7IDddDA" outputId="c3997578-7020-4ffb-81fe-8220cf5ff5e8"}
``` {.python}
# Encoding the categorical variables

##Every column having more than two unique values will be splitted in the number of values that column have
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
features.head(5)
```

::: {.output .execute_result execution_count="9"}
```{=html}
  <div id="df-ae9ed1e4-84b6-44ec-b4f7-a5da24983d03">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>40</td>
      <td>140</td>
      <td>289</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>0.0</td>
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
      <td>49</td>
      <td>160</td>
      <td>180</td>
      <td>0</td>
      <td>156</td>
      <td>0</td>
      <td>1.0</td>
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
      <td>37</td>
      <td>130</td>
      <td>283</td>
      <td>0</td>
      <td>98</td>
      <td>0</td>
      <td>0.0</td>
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
      <td>48</td>
      <td>138</td>
      <td>214</td>
      <td>0</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
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
      <td>54</td>
      <td>150</td>
      <td>195</td>
      <td>0</td>
      <td>122</td>
      <td>0</td>
      <td>0.0</td>
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-ae9ed1e4-84b6-44ec-b4f7-a5da24983d03')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ae9ed1e4-84b6-44ec-b4f7-a5da24983d03 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ae9ed1e4-84b6-44ec-b4f7-a5da24983d03');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code colab="{\"height\":270,\"base_uri\":\"https://localhost:8080/\"}" id="gQrtZsM_hEgU" outputId="98a627d3-1c13-4568-9ca4-b349e2d7d3b9"}
``` {.python}
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

::: {.output .execute_result execution_count="10"}
```{=html}
  <div id="df-aa71f92f-1eb9-4529-9710-69929d324763">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-aa71f92f-1eb9-4529-9710-69929d324763')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-aa71f92f-1eb9-4529-9710-69929d324763 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-aa71f92f-1eb9-4529-9710-69929d324763');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="5A1lLg_y5IOW"}
The features having continuous values are scaled are between 0 and 1.
The features having categorical values are ecnoded with 0 and 1. Now the
dataset is ready for preparing the model.
:::

::: {.cell .markdown id="CPTag86F5miy"}
# Classifier Modeling
:::

::: {.cell .markdown id="ZWLZB1AJ5wUI"}
## Coverting the data into matrix form for performing the mathematical operations
:::

::: {.cell .markdown id="Sfhp-oLY6NXY"}
It is known that the hypothesis function for Logistic regression
classifier is

\$ h(X) = \\frac{1}{1 + e\^{-Z}}\$, where $Z$ = $X\theta$

Let the dataset has $m$ features and $n$ samples then,

1.  $h(X)$ -\> Vector of size $n$
2.  $\theta$ is a constant vector with $m+1$ co-ordinates
3.  $Z$ -\> Vector of size $n$
4.  $X$ is the features matrix where the first column has entry 1 in
    every row therefore the size of $X$ is $n \times (m+1)$
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="DncpfouZ_fpT" outputId="94ab6382-c663-41ff-e4b6-a6bb3c82917b"}
``` {.python}
n = features.shape[0] #number of samples
m = features.shape[1] #number of features
features_mat = features.to_numpy() 
X = np.hstack((np.ones((n, 1)), features_mat)) #adding column of 1
print("X", "\n", X[0:3, :])

Y = target.to_numpy()
print("Y", '\n', Y[:10])
```

::: {.output .stream .stdout}
    X 
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
    Y 
     [0 1 0 1 0 0 0 0 1 0]
:::
:::

::: {.cell .markdown id="_C6MFWA5Bl-o"}
### Splittig the data for train and test
:::

::: {.cell .code id="tBLX4uENBsCF"}
``` {.python}
ntrain = round(0.8*n) #test_size = 0.2
xtrain = X[0:ntrain, :]  
xtest = X[ntrain:n+1,:]

ytrain = Y[0:ntrain] 
ytest = Y[ntrain:]
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ssJkme-Cd0T5" outputId="b7010c13-8c9b-4fad-e514-990fd9ff66cb"}
``` {.python}
np.size(ytrain)
```

::: {.output .execute_result execution_count="30"}
    734
:::
:::

::: {.cell .markdown id="yqz2aiCxCknD"}
## Functions involved in classifier
:::

::: {.cell .markdown id="bmPB9pGoEBcJ"}
1.  Hypothesis function: $h(x) = \frac{1}{1 + e^{-z}}$ \--\> predicted
    value
2.  $Z = X\theta$
3.  Cost Function: $J(\theta) = -\frac{1}{n}[log(1 - h(x)) + yz]$, where
    $y$ is actual value
4.  As the minimization of the cost function is not a simple function,
    either Newton Raphson or the gredient descent method would be used
    to solve.
5.  Gredient function: $\nabla _\theta J(\theta) = x^T(y - h(x))$
6.  Getting the optimized theta:
    $\theta _{new} = \theta _{old} - \eta \{x^T(y-h(x))\}$, where $\eta$
    is learing rate
:::

::: {.cell .code id="XY6EB9C3C0n3"}
``` {.python}
# Hypothesis function
def sigma(xtrain,theta):
  z = np.matmul(xtrain, theta)
  ze = np.exp(-z)
  hx = 1 / (1 + ze)
  return(z, hx)

#Cost function 
def costf(z,hx,ytrain):
  c1 = np.matmul(ytrain, np.log(hx))
  c2 = np.matmul((1-ytrain), np.log(1-hx))
  cost = -(c1+c2)/n
  return(cost)
  

#Optimization of cost function
def Optimize(xtrain,ytrain,repeat,eta):
  n = np.shape(xtrain)[0]
  theta = np.random.random(np.shape(xtrain)[1]) #initialization of theta
  cost_list = []                           #list of cost calculated for a learning rate for visualizing the variation of cost 
  for i in range(repeat):
    (z, hx) = sigma(xtrain,theta)
    nabla = np.matmul(xtrain.transpose(), (hx-ytrain))/n #the gredient
    theta = theta - eta*nabla
    cost_list.append(costf(z,hx,ytrain))
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
def classifier(xtrain,ytrain,eta_list,xtest,ytest,repeat):
  reports = pd.DataFrame(columns = ['eta', 'accuracy', 'precision', 'recall', 'F-1 Score'])
  predicted_values = pd.DataFrame()
  predicted_values['Actual label'] = ytest
  cost_variation = {}
  theta_variation = {}
  for eta in eta_list:
    (theta, cost_list)  = Optimize(xtrain,ytrain,repeat,eta)
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
:::

::: {.cell .code id="oKr4WRdQgtKQ"}
``` {.python}
eta_list = [0.0001, 0.001, 0.01, 0.1]
(cost_variation, reports, predicted_values,theta_variation) = classifier(xtrain, ytrain, eta_list, xtest, ytest, 1000)
```
:::

::: {.cell .code id="iH0ksuru1krE"}
``` {.python}
fig = plt.gcf();
fig.set_size_inches(8,6)
for key in cost_variation.keys():
  y = cost_variation[0.01]
  x = list(range(len(y)))
  sns.lineplot(x,y,label=key)
  plt.show()
```
:::

::: {.cell .code colab="{\"height\":174,\"base_uri\":\"https://localhost:8080/\"}" id="Wyfpb8pa2xLj" outputId="5d16e8ec-e4c3-4f2a-a7db-029116df7584"}
``` {.python}
reports
```

::: {.output .execute_result execution_count="87"}
```{=html}
  <div id="df-9304c062-c313-461c-92f3-f078d039078e">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eta</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>F-1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0001</td>
      <td>0.467391</td>
      <td>0.467391</td>
      <td>1.000000</td>
      <td>0.637037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0010</td>
      <td>0.467391</td>
      <td>0.467391</td>
      <td>1.000000</td>
      <td>0.637037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0100</td>
      <td>0.750000</td>
      <td>0.700000</td>
      <td>0.813953</td>
      <td>0.752688</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1000</td>
      <td>0.771739</td>
      <td>0.755814</td>
      <td>0.755814</td>
      <td>0.755814</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9304c062-c313-461c-92f3-f078d039078e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9304c062-c313-461c-92f3-f078d039078e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9304c062-c313-461c-92f3-f078d039078e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code colab="{\"height\":138,\"base_uri\":\"https://localhost:8080/\"}" id="IuQFlzjk0u5T" outputId="9a9cf3b0-a0c9-41fc-ab58-d8168b5ab841"}
``` {.python}
eta_list = [ 0.001, 0.01, 0.1]
classifier(xtrain,ytrain,xtest,ytest,1000,eta_list)
```

::: {.output .stream .stdout}
    eta =  0.001 
     {'accuracy': 0.7717391304347826, 'precision': 0.7558139534883721, 'Recall': 0.7558139534883721, 'F-1 Score': 0.755813953488372}
    eta =  0.01 
     {'accuracy': 0.7771739130434783, 'precision': 0.7528089887640449, 'Recall': 0.7790697674418605, 'F-1 Score': 0.7657142857142858}
    eta =  0.1 
     {'accuracy': 0.7771739130434783, 'precision': 0.7777777777777778, 'Recall': 0.7325581395348837, 'F-1 Score': 0.7544910179640719}
:::

::: {.output .display_data}
    <Figure size 720x432 with 0 Axes>
:::
:::

::: {.cell .code id="XT77dEnvcQwj"}
``` {.python}
def costf(z,hx,ytrain):
  ze = np.exp(z)
  n = np.size(z)
  hxx = 1 / (1 + ze)
  cost = np.sum((np.log(hxx) + np.matmul(ytrain, z))/n)
  return(cost)
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="HzhXkPYpcWcv" outputId="f44ff855-d937-46f9-d484-c0674add124f"}
``` {.python}
for eta in eta_list:
  print(eta,'\n',theta_variation[eta])
```

::: {.output .stream .stdout}
    0.0001 
     [0.13349684 0.47694926 0.43077015 0.52900916 0.48727437 0.26937076
     0.23072645 0.00924484 0.61460649 0.50958201 0.82342853 0.29419408
     0.34490218 0.05078802 0.66607007 0.16807083 0.88205953 0.43074446
     0.07643548 0.83354775]
    0.001 
     [0.11835009 0.21368097 0.58354663 0.81176857 0.10913029 0.37737561
     0.62274339 0.0669014  0.11756649 0.25770441 0.00958604 0.82504812
     0.19794057 0.57433793 0.44230805 0.47253602 0.446914   0.85548406
     0.20022754 0.21266521]
    0.01 
     [-0.85283432  0.18676266 -0.47008446  0.06298485  0.51365889 -0.04134847
      0.8058665   0.46953883 -0.48708716 -0.21609037  0.72749412  0.12102221
      0.20783877  0.34480201  0.65319209 -0.90482245  0.47435321  0.21309659
      0.45313793  0.24421899]
    0.1 
     [-0.3409352  -0.05617829 -0.03456318 -0.8920723   1.51032323 -0.90232064
      0.86431808  0.90235451 -0.74694836 -0.1237637   1.23023006  0.3375241
     -0.32851373 -0.22310954 -0.13899839 -1.43491515  1.35586258  0.19553387
      0.38180144 -0.79421407]
    1 
     [-0.70686306  0.44682114  0.31856058 -2.55160511  1.6110569  -0.85191257
      0.81043707  1.45116236 -0.76555174 -0.20621917  1.24847406  0.27327992
      0.01222114 -0.05063459  0.2430166  -1.58712206  1.37432028 -0.15008021
      0.32694027 -0.83658276]
:::
:::

::: {.cell .code colab="{\"height\":174,\"base_uri\":\"https://localhost:8080/\"}" id="YLNK0zSXdctU" outputId="0f65e7bd-4ca3-48d3-a7e3-248de5678f0e"}
``` {.python}
reports
```

::: {.output .execute_result execution_count="32"}
```{=html}
  <div id="df-68c735d6-f67f-454b-8a19-fb36d6ad6731">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eta</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>F-1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0001</td>
      <td>0.467391</td>
      <td>0.467391</td>
      <td>1.000000</td>
      <td>0.637037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0010</td>
      <td>0.467391</td>
      <td>0.467391</td>
      <td>1.000000</td>
      <td>0.637037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0100</td>
      <td>0.728261</td>
      <td>0.666667</td>
      <td>0.837209</td>
      <td>0.742268</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1000</td>
      <td>0.777174</td>
      <td>0.758621</td>
      <td>0.767442</td>
      <td>0.763006</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-68c735d6-f67f-454b-8a19-fb36d6ad6731')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-68c735d6-f67f-454b-8a19-fb36d6ad6731 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-68c735d6-f67f-454b-8a19-fb36d6ad6731');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="F1UqyMSUhc2k" outputId="0d7a5135-f45c-427d-d908-d731cafd2862"}
``` {.python}
np.matmul(ytrain,z)
```

::: {.output .execute_result execution_count="16"}
    1940.6362736316646
:::
:::

::: {.cell .code id="9mXrlyQZdSki"}
``` {.python}
cost = costf(z,hx,ytrain)
```
:::

::: {.cell .code id="urVTkjtvi2PD"}
``` {.python}
#Using Scikit learn
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
cm = confusion_matrix(ytest, ypred)
report = classification_report(ytest, ypred)
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ID7b3-usjBfj" outputId="4274dfdc-3872-464c-e521-b433d80f2988"}
``` {.python}
print(report)
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.82      0.77      0.79       113
               1       0.85      0.88      0.86       163

        accuracy                           0.84       276
       macro avg       0.83      0.83      0.83       276
    weighted avg       0.84      0.84      0.84       276
:::
:::

::: {.cell .code id="RIocDar_vE7v"}
``` {.python}
eta_list = [0.1]
(cost_variation, reports, predicted_values,theta_variation) = classifier(xtrain, ytrain, eta_list, xtest, ytest, 1000)
```
:::

::: {.cell .code colab="{\"height\":206,\"base_uri\":\"https://localhost:8080/\"}" id="mSwqY9_zwu4l" outputId="1bbd7c73-1371-4bd5-de4d-fd0bc6532da9"}
``` {.python}
reports.transpose()
```

::: {.output .execute_result execution_count="90"}
```{=html}
  <div id="df-bc876616-7ae7-4870-9a24-d0b22450b3e9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>eta</th>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.771739</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.755814</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.755814</td>
    </tr>
    <tr>
      <th>F-1 Score</th>
      <td>0.755814</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bc876616-7ae7-4870-9a24-d0b22450b3e9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bc876616-7ae7-4870-9a24-d0b22450b3e9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bc876616-7ae7-4870-9a24-d0b22450b3e9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code id="t0ocJqgFw-HN"}
``` {.python}
```
:::

::: {.cell .code colab="{\"height\":279,\"base_uri\":\"https://localhost:8080/\"}" id="gaNkYu7jxuDW" outputId="d1d1d297-4e73-454c-f477-cfac71d14559"}
``` {.python}
#Visulaizing the prediction
df_pre = df.iloc[ntrain:]
df_pre = df_pre[['Age', 'Cholesterol']]
df_pre['Actual label'] = list(predicted_values['Actual label'])
df_pre['predicted'] = list(predicted_values['label_0.1'])

true = df_pre[df_pre['Actual label'] == df_pre['predicted']]
false = df_pre[df_pre['Actual label'] != df_pre['predicted']]
sns.scatterplot(x = 'Age', y = 'Cholesterol', data = true, marker = '*', color = 'blue', label = 'predicted_true');
sns.scatterplot(x = 'Age', y = 'Cholesterol', data = false, marker = '+', color = 'yellow', label = 'predicted_false');
plt.show()
```

::: {.output .display_data}
![](vertopal_94df26f0b5b0412b8d0ec190e0fd7a31/80a582a6140da31c57bca53af63e3792b95591c7.png)
:::
:::

::: {.cell .code colab="{\"height\":423,\"base_uri\":\"https://localhost:8080/\"}" id="MWIhJLHb2wN6" outputId="c8700e8a-c3ef-4250-dbdc-e9404df6114c"}
``` {.python}
false
```

::: {.output .execute_result execution_count="108"}
```{=html}
  <div id="df-decf01cf-8038-4f43-8bd9-2a17e6c8c785">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cholesterol</th>
      <th>Actual label</th>
      <th>predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>734</th>
      <td>56</td>
      <td>283</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>735</th>
      <td>49</td>
      <td>188</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>736</th>
      <td>54</td>
      <td>286</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>737</th>
      <td>57</td>
      <td>274</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>738</th>
      <td>65</td>
      <td>360</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>913</th>
      <td>45</td>
      <td>264</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>914</th>
      <td>68</td>
      <td>193</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>915</th>
      <td>57</td>
      <td>131</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>916</th>
      <td>57</td>
      <td>236</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>917</th>
      <td>38</td>
      <td>175</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>184 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-decf01cf-8038-4f43-8bd9-2a17e6c8c785')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-decf01cf-8038-4f43-8bd9-2a17e6c8c785 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-decf01cf-8038-4f43-8bd9-2a17e6c8c785');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code colab="{\"height\":423,\"base_uri\":\"https://localhost:8080/\"}" id="JpsXPUMc3yG-" outputId="5bc5615f-2c5d-49e5-9bde-f0934f6bfbda"}
``` {.python}
predicted_values
```

::: {.output .execute_result execution_count="109"}
```{=html}
  <div id="df-081e865d-d60b-4ea2-83e7-3b13956467d1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual label</th>
      <th>label_0.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>180</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>181</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>182</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>183</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>184 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-081e865d-d60b-4ea2-83e7-3b13956467d1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-081e865d-d60b-4ea2-83e7-3b13956467d1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-081e865d-d60b-4ea2-83e7-3b13956467d1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::
