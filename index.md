## Introduction
- This is project about creating self Logistic regression classifier using python and Comparing the performance with the model created using the inbuilt library scikit learn.

- The dataset for this procject is "Heart Failure Prediction Dataset" downloaded from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

- The outflow of the project is as:

    a. Overview of the data 

    b. Visualization

    c. Data Processing

    d. Classifier Modeling

    e. Classifier using Scikit learn

    f. Comparision and coclustion
  
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
<html>
    <a href="#" class="image fit"><img src="/MSLG_df.head5.png" alt="Top five rows of the dataset" /></a>
</html>

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

<html>
    <a href="#" class="image fit"><img src="/MSLG_df.head5.png" alt="" /></a>
</html>
For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/rahulguptta/rahulguptta.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
