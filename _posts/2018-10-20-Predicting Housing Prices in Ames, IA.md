---
layout: post
title: "Ames Housing Project"
date: 2018-09-18
excerpt: "What factors are most predictive of the price of a house in Ames Iowa?"
project: true
tags: [project, regression, regularization, interaction terms]
image: "/assets/img/iowa_house.jpg"
---

```python
# imports that I'll use over the course of the project
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score

%matplotlib inline
```


```python
# read in data
houses = pd.read_csv('./data/train.csv')
holdout_set = pd.read_csv('./data/test.csv')
```

# Basic EDA


```python
# Check out the data types
houses.dtypes
```




    Id                  int64
    PID                 int64
    MS SubClass         int64
    MS Zoning          object
    Lot Frontage      float64
    Lot Area            int64
    Street             object
    Alley              object
    Lot Shape          object
    Land Contour       object
    Utilities          object
    Lot Config         object
    Land Slope         object
    Neighborhood       object
    Condition 1        object
    Condition 2        object
    Bldg Type          object
    House Style        object
    Overall Qual        int64
    Overall Cond        int64
    Year Built          int64
    Year Remod/Add      int64
    Roof Style         object
    Roof Matl          object
    Exterior 1st       object
    Exterior 2nd       object
    Mas Vnr Type       object
    Mas Vnr Area      float64
    Exter Qual         object
    Exter Cond         object
                       ...   
    Half Bath           int64
    Bedroom AbvGr       int64
    Kitchen AbvGr       int64
    Kitchen Qual       object
    TotRms AbvGrd       int64
    Functional         object
    Fireplaces          int64
    Fireplace Qu       object
    Garage Type        object
    Garage Yr Blt     float64
    Garage Finish      object
    Garage Cars       float64
    Garage Area       float64
    Garage Qual        object
    Garage Cond        object
    Paved Drive        object
    Wood Deck SF        int64
    Open Porch SF       int64
    Enclosed Porch      int64
    3Ssn Porch          int64
    Screen Porch        int64
    Pool Area           int64
    Pool QC            object
    Fence              object
    Misc Feature       object
    Misc Val            int64
    Mo Sold             int64
    Yr Sold             int64
    Sale Type          object
    SalePrice           int64
    Length: 81, dtype: object




```python
# Basic description to check any weirdness
houses.describe().T
```




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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Id</th>
      <td>2051.0</td>
      <td>1.474034e+03</td>
      <td>8.439808e+02</td>
      <td>1.0</td>
      <td>753.5</td>
      <td>1486.0</td>
      <td>2.198000e+03</td>
      <td>2930.0</td>
    </tr>
    <tr>
      <th>PID</th>
      <td>2051.0</td>
      <td>7.135900e+08</td>
      <td>1.886918e+08</td>
      <td>526301100.0</td>
      <td>528458140.0</td>
      <td>535453200.0</td>
      <td>9.071801e+08</td>
      <td>924152030.0</td>
    </tr>
    <tr>
      <th>MS SubClass</th>
      <td>2051.0</td>
      <td>5.700878e+01</td>
      <td>4.282422e+01</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>7.000000e+01</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>Lot Frontage</th>
      <td>1721.0</td>
      <td>6.905520e+01</td>
      <td>2.326065e+01</td>
      <td>21.0</td>
      <td>58.0</td>
      <td>68.0</td>
      <td>8.000000e+01</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>Lot Area</th>
      <td>2051.0</td>
      <td>1.006521e+04</td>
      <td>6.742489e+03</td>
      <td>1300.0</td>
      <td>7500.0</td>
      <td>9430.0</td>
      <td>1.151350e+04</td>
      <td>159000.0</td>
    </tr>
    <tr>
      <th>Overall Qual</th>
      <td>2051.0</td>
      <td>6.112140e+00</td>
      <td>1.426271e+00</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.000000e+00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Overall Cond</th>
      <td>2051.0</td>
      <td>5.562165e+00</td>
      <td>1.104497e+00</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.000000e+00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Year Built</th>
      <td>2051.0</td>
      <td>1.971709e+03</td>
      <td>3.017789e+01</td>
      <td>1872.0</td>
      <td>1953.5</td>
      <td>1974.0</td>
      <td>2.001000e+03</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>Year Remod/Add</th>
      <td>2051.0</td>
      <td>1.984190e+03</td>
      <td>2.103625e+01</td>
      <td>1950.0</td>
      <td>1964.5</td>
      <td>1993.0</td>
      <td>2.004000e+03</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>Mas Vnr Area</th>
      <td>2029.0</td>
      <td>9.969591e+01</td>
      <td>1.749631e+02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.610000e+02</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>BsmtFin SF 1</th>
      <td>2050.0</td>
      <td>4.423005e+02</td>
      <td>4.612041e+02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>368.0</td>
      <td>7.337500e+02</td>
      <td>5644.0</td>
    </tr>
    <tr>
      <th>BsmtFin SF 2</th>
      <td>2050.0</td>
      <td>4.795902e+01</td>
      <td>1.650009e+02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1474.0</td>
    </tr>
    <tr>
      <th>Bsmt Unf SF</th>
      <td>2050.0</td>
      <td>5.677283e+02</td>
      <td>4.449548e+02</td>
      <td>0.0</td>
      <td>220.0</td>
      <td>474.5</td>
      <td>8.110000e+02</td>
      <td>2336.0</td>
    </tr>
    <tr>
      <th>Total Bsmt SF</th>
      <td>2050.0</td>
      <td>1.057988e+03</td>
      <td>4.494107e+02</td>
      <td>0.0</td>
      <td>793.0</td>
      <td>994.5</td>
      <td>1.318750e+03</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <th>1st Flr SF</th>
      <td>2051.0</td>
      <td>1.164488e+03</td>
      <td>3.964469e+02</td>
      <td>334.0</td>
      <td>879.5</td>
      <td>1093.0</td>
      <td>1.405000e+03</td>
      <td>5095.0</td>
    </tr>
    <tr>
      <th>2nd Flr SF</th>
      <td>2051.0</td>
      <td>3.293291e+02</td>
      <td>4.256710e+02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.925000e+02</td>
      <td>1862.0</td>
    </tr>
    <tr>
      <th>Low Qual Fin SF</th>
      <td>2051.0</td>
      <td>5.512921e+00</td>
      <td>5.106887e+01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1064.0</td>
    </tr>
    <tr>
      <th>Gr Liv Area</th>
      <td>2051.0</td>
      <td>1.499330e+03</td>
      <td>5.004478e+02</td>
      <td>334.0</td>
      <td>1129.0</td>
      <td>1444.0</td>
      <td>1.728500e+03</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>Bsmt Full Bath</th>
      <td>2049.0</td>
      <td>4.275256e-01</td>
      <td>5.226732e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Bsmt Half Bath</th>
      <td>2049.0</td>
      <td>6.344558e-02</td>
      <td>2.517052e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Full Bath</th>
      <td>2051.0</td>
      <td>1.577279e+00</td>
      <td>5.492794e-01</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.000000e+00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Half Bath</th>
      <td>2051.0</td>
      <td>3.710385e-01</td>
      <td>5.010427e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Bedroom AbvGr</th>
      <td>2051.0</td>
      <td>2.843491e+00</td>
      <td>8.266183e-01</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.000000e+00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Kitchen AbvGr</th>
      <td>2051.0</td>
      <td>1.042906e+00</td>
      <td>2.097900e-01</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000e+00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>TotRms AbvGrd</th>
      <td>2051.0</td>
      <td>6.435885e+00</td>
      <td>1.560225e+00</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.000000e+00</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>2051.0</td>
      <td>5.909313e-01</td>
      <td>6.385163e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.000000e+00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Garage Yr Blt</th>
      <td>1937.0</td>
      <td>1.978708e+03</td>
      <td>2.544109e+01</td>
      <td>1895.0</td>
      <td>1961.0</td>
      <td>1980.0</td>
      <td>2.002000e+03</td>
      <td>2207.0</td>
    </tr>
    <tr>
      <th>Garage Cars</th>
      <td>2050.0</td>
      <td>1.776585e+00</td>
      <td>7.645374e-01</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.000000e+00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Garage Area</th>
      <td>2050.0</td>
      <td>4.736717e+02</td>
      <td>2.159346e+02</td>
      <td>0.0</td>
      <td>319.0</td>
      <td>480.0</td>
      <td>5.760000e+02</td>
      <td>1418.0</td>
    </tr>
    <tr>
      <th>Wood Deck SF</th>
      <td>2051.0</td>
      <td>9.383374e+01</td>
      <td>1.285494e+02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.680000e+02</td>
      <td>1424.0</td>
    </tr>
    <tr>
      <th>Open Porch SF</th>
      <td>2051.0</td>
      <td>4.755680e+01</td>
      <td>6.674724e+01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>7.000000e+01</td>
      <td>547.0</td>
    </tr>
    <tr>
      <th>Enclosed Porch</th>
      <td>2051.0</td>
      <td>2.257192e+01</td>
      <td>5.984511e+01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>432.0</td>
    </tr>
    <tr>
      <th>3Ssn Porch</th>
      <td>2051.0</td>
      <td>2.591419e+00</td>
      <td>2.522961e+01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>508.0</td>
    </tr>
    <tr>
      <th>Screen Porch</th>
      <td>2051.0</td>
      <td>1.651146e+01</td>
      <td>5.737420e+01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>490.0</td>
    </tr>
    <tr>
      <th>Pool Area</th>
      <td>2051.0</td>
      <td>2.397855e+00</td>
      <td>3.778257e+01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>800.0</td>
    </tr>
    <tr>
      <th>Misc Val</th>
      <td>2051.0</td>
      <td>5.157435e+01</td>
      <td>5.733940e+02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>Mo Sold</th>
      <td>2051.0</td>
      <td>6.219893e+00</td>
      <td>2.744736e+00</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>8.000000e+00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>Yr Sold</th>
      <td>2051.0</td>
      <td>2.007776e+03</td>
      <td>1.312014e+00</td>
      <td>2006.0</td>
      <td>2007.0</td>
      <td>2008.0</td>
      <td>2.009000e+03</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>2051.0</td>
      <td>1.814697e+05</td>
      <td>7.925866e+04</td>
      <td>12789.0</td>
      <td>129825.0</td>
      <td>162500.0</td>
      <td>2.140000e+05</td>
      <td>611657.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Understand the size of the data
houses.shape
```




    (2051, 81)




```python
# Null value check
df_nulls = pd.DataFrame(data=houses.isnull().sum(), columns=['Nulls'])
df_nulls.sort_values('Nulls', ascending=False)
# There seems to be quite a few null values, should look at cleaning these up.
```




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
      <th>Nulls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pool QC</th>
      <td>2042</td>
    </tr>
    <tr>
      <th>Misc Feature</th>
      <td>1986</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1911</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1651</td>
    </tr>
    <tr>
      <th>Fireplace Qu</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>Lot Frontage</th>
      <td>330</td>
    </tr>
    <tr>
      <th>Garage Finish</th>
      <td>114</td>
    </tr>
    <tr>
      <th>Garage Qual</th>
      <td>114</td>
    </tr>
    <tr>
      <th>Garage Yr Blt</th>
      <td>114</td>
    </tr>
    <tr>
      <th>Garage Cond</th>
      <td>114</td>
    </tr>
    <tr>
      <th>Garage Type</th>
      <td>113</td>
    </tr>
    <tr>
      <th>Bsmt Exposure</th>
      <td>58</td>
    </tr>
    <tr>
      <th>BsmtFin Type 2</th>
      <td>56</td>
    </tr>
    <tr>
      <th>BsmtFin Type 1</th>
      <td>55</td>
    </tr>
    <tr>
      <th>Bsmt Cond</th>
      <td>55</td>
    </tr>
    <tr>
      <th>Bsmt Qual</th>
      <td>55</td>
    </tr>
    <tr>
      <th>Mas Vnr Area</th>
      <td>22</td>
    </tr>
    <tr>
      <th>Mas Vnr Type</th>
      <td>22</td>
    </tr>
    <tr>
      <th>Bsmt Half Bath</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Bsmt Full Bath</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Garage Area</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Total Bsmt SF</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Bsmt Unf SF</th>
      <td>1</td>
    </tr>
    <tr>
      <th>BsmtFin SF 2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>BsmtFin SF 1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Garage Cars</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Mo Sold</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Sale Type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Full Bath</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Half Bath</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>MS Zoning</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Lot Area</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Street</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Lot Shape</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Land Contour</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Lot Config</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Land Slope</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bldg Type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>House Style</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Overall Cond</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2nd Flr SF</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Year Built</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Year Remod/Add</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Roof Style</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Roof Matl</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Exterior 1st</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Exterior 2nd</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Exter Qual</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Exter Cond</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Foundation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>PID</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Heating QC</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Central Air</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1st Flr SF</th>
      <td>0</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>81 rows × 1 columns</p>
</div>




```python
# If and when I wanna drop all na's from original DF:
# houses.dropna(axis=0, how='any')
# houses.select_dtypes(object)
```

# Feature Engineering

# Create Dummy Variables


```python
# Setup dummy variables for later
houses_object_columns = pd.get_dummies(houses,columns=houses.select_dtypes(object).columns)
```


```python
houses_object_columns.head()
```




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
      <th>Id</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Mas Vnr Area</th>
      <th>...</th>
      <th>Misc Feature_TenC</th>
      <th>Sale Type_COD</th>
      <th>Sale Type_CWD</th>
      <th>Sale Type_Con</th>
      <th>Sale Type_ConLD</th>
      <th>Sale Type_ConLI</th>
      <th>Sale Type_ConLw</th>
      <th>Sale Type_New</th>
      <th>Sale Type_Oth</th>
      <th>Sale Type_WD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>109</td>
      <td>533352170</td>
      <td>60</td>
      <td>NaN</td>
      <td>13517</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>2005</td>
      <td>289.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>544</td>
      <td>531379050</td>
      <td>60</td>
      <td>43.0</td>
      <td>11492</td>
      <td>7</td>
      <td>5</td>
      <td>1996</td>
      <td>1997</td>
      <td>132.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>153</td>
      <td>535304180</td>
      <td>20</td>
      <td>68.0</td>
      <td>7922</td>
      <td>5</td>
      <td>7</td>
      <td>1953</td>
      <td>2007</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>318</td>
      <td>916386060</td>
      <td>60</td>
      <td>73.0</td>
      <td>9802</td>
      <td>5</td>
      <td>5</td>
      <td>2006</td>
      <td>2007</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255</td>
      <td>906425045</td>
      <td>50</td>
      <td>82.0</td>
      <td>14235</td>
      <td>6</td>
      <td>8</td>
      <td>1900</td>
      <td>1993</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 292 columns</p>
</div>




```python
df_nulls_obj = pd.DataFrame(data=houses_object_columns.isnull().sum(), columns=['Nulls'])
df_nulls_obj.sort_values('Nulls', ascending=False)
```




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
      <th>Nulls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lot Frontage</th>
      <td>330</td>
    </tr>
    <tr>
      <th>Garage Yr Blt</th>
      <td>114</td>
    </tr>
    <tr>
      <th>Mas Vnr Area</th>
      <td>22</td>
    </tr>
    <tr>
      <th>Bsmt Half Bath</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Bsmt Full Bath</th>
      <td>2</td>
    </tr>
    <tr>
      <th>BsmtFin SF 1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Garage Area</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Total Bsmt SF</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Bsmt Unf SF</th>
      <td>1</td>
    </tr>
    <tr>
      <th>BsmtFin SF 2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Garage Cars</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Bsmt Exposure_No</th>
      <td>0</td>
    </tr>
    <tr>
      <th>BsmtFin Type 1_ALQ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>BsmtFin Type 1_BLQ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>BsmtFin Type 1_GLQ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Exposure_Gd</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Exposure_Mn</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Exposure_Av</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Cond_TA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Cond_Po</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Cond_Fa</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Cond_Ex</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Qual_TA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Qual_Po</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Qual_Gd</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Qual_Fa</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bsmt Cond_Gd</th>
      <td>0</td>
    </tr>
    <tr>
      <th>BsmtFin Type 1_Unf</th>
      <td>0</td>
    </tr>
    <tr>
      <th>BsmtFin Type 1_LwQ</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Neighborhood_Timber</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_StoneBr</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_Somerst</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_SawyerW</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_SWISU</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bldg Type_2fmCon</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_OldTown</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_NridgHt</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_NoRidge</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_NWAmes</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_NPkVill</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Neighborhood_NAmes</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 1_Feedr</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 1_Norm</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 1_PosA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 1_PosN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 1_RRAe</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 1_RRAn</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 1_RRNe</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 1_RRNn</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 2_Artery</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 2_Feedr</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 2_Norm</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 2_PosA</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 2_PosN</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 2_RRAe</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 2_RRAn</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Condition 2_RRNn</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Bldg Type_1Fam</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Sale Type_WD</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>292 rows × 1 columns</p>
</div>




```python
# Clean up NAN
houses_object_columns.fillna(value=0, inplace=True)
```

# First run of features


```python
# Most of the code after the features section was run multiple times to attempt to find the best model
```


```python
# Original run of feature engineering
# Just randomly picking a few potential predictors
features = ['Lot Area', 'Overall Qual','Fireplaces','TotRms AbvGrd']
X = houses[features]
y = houses['SalePrice']
```

# Second run of features


```python
# I'm going to pull out only numeric columns, to see if there's some variables with high correlation that I can throw in to improve my model.
# Make it into it's own DF to run some EDA on it
houses_numonly = houses.select_dtypes(np.number)
```


```python
# houses_numonly_dropna = houses_numonly.dropna(axis=0, how='any')
# houses_numonly_dropna.isnull().sum()
```


```python
# Original run
# houses_numonly.columns
houses_numonly.shape
```


```python
# For original run of determining correlations
features = houses_numonly.columns
X = houses[features]
y = houses_numonly['SalePrice']
```


```python
# X.corr() is far too painful to look at here
# Run the heatmap, see if anything sticks out
plt.subplots(figsize=(40,30))
sns.heatmap(X.corr(), annot=True)
```


```python
# Just eyeballing it, some important correlations I see (descending):
# I stopped at 'Garage Yr Blt', which had a correlation of .53
# Upon further review, 'Garage Yr Blt' had far too many missing data points, so was left off
features = ['Overall Qual',
'Gr Liv Area',
'Garage Cars',
'Garage Area',
'Total Bsmt SF',
'1st Flr SF',
'Year Built',
'Year Remod/Add',
'Full Bath']
# 'Garage Yr Blt']
```

# Third run of features


```python
# Third time around: going to add a few variables - might cause overfit, but let's see
# Was more lax on what variables I added (based on correlation > .3)
features = ['Overall Qual',
'Gr Liv Area',
'Garage Cars',
'Garage Area',
'Total Bsmt SF',
'1st Flr SF',
'Year Built',
'Year Remod/Add',
'Full Bath',
'TotRms AbvGrd',
'Mas Vnr Area',
'Fireplaces',
'BsmtFin SF 1',
'Wood Deck SF']
```

# Code below was run on multiple sets of features


```python
# Set your in/dependent variables
# Original usage
# X = houses[features]
# y = houses['SalePrice']
```


```python
# Old run:
# X = houses_object_columns[features]
# y = houses_object_columns['SalePrice']
```


```python
# Null check
X.isnull().sum()
```


```python
# Fill in the nas:
# Despite the warning, this still works:
X['Mas Vnr Area'].fillna(value=0, inplace=True)
X['Garage Cars'].fillna(value=0, inplace=True)
X['Garage Area'].fillna(value=0, inplace=True)
X['Total Bsmt SF'].fillna(value=0, inplace=True)
X['BsmtFin SF 1'].fillna(value=0, inplace=True)
```


```python
# Check:
X.isnull().sum()
```


```python
# X.shape[0]
```

# Graveyard for data cleanup


```python
# 2nd feature run note: Going to rebuild my features, minus Grg yr blt as there seems to be more than a few nulls
# Some of the next few cells were only used for original run through of cleaning up features
# Take a look at null values in my features:
# X[X['Garage Cars'].isnull() == True]
# Looks like the singular null in 'Grg Cars' is one in the same w/ 'Grg Area'
```


```python
# # Dropping the rows with the null values
# X = X.drop(X.index[1712])
# # Sets have to match in row length:
# y = y.drop(y.index[1712])
```


```python
# Check to make sure it worked
# X[X['Garage Cars'].isnull() == True]
```


```python
# X[X['Total Bsmt SF'].isnull() == True]
```


```python
# Dropping the rows with the null values
# X = X.drop(X.index[1327])
# Sets have to match in row length:
# y = y.drop(y.index[1327])
```


```python
# Check to make sure it worked
# X[X['Total Bsmt SF'].isnull() == True]
```


```python
# Messing around trying to figure out how to locate the null rows
# X.columns[X.isna().any()].tolist()
# X.loc[:,X.isna().any()]
```

# Lasso


```python
X = houses_object_columns.drop('SalePrice', axis=1)
y = houses_object_columns['SalePrice']
```


```python
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
```


```python
X_train.fillna(value=0, inplace=True)
# X_test.fillna(value=0, inplace=True)
y_train.fillna(value=0, inplace=True)
# y_test.fillna(value=0, inplace=True)
```

```python
L = Lasso(alpha = 2.5, max_iter=10000, random_state = 42)
```


```python
L.fit(X_train, y_train)
```




    Lasso(alpha=2.5, copy_X=True, fit_intercept=True, max_iter=10000,
       normalize=False, positive=False, precompute=False, random_state=42,
       selection='cyclic', tol=0.0001, warm_start=False)




```python
# Original attempts:
# LinearRegression
# linreg = LinearRegression()
# Original fit
# linreg.fit(X, y)
# linreg.coef_
```


```python
# Using KFold to help randomize the folds, and also see if different splits get me significantly different scores
kf = KFold(n_splits=5, random_state=42, shuffle=True)
```


```python
scores = cross_val_score(L, X_train, y_train, cv=kf)
print(scores)
print(scores.mean())
```

    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    [0.91818856 0.93117813 0.9224465  0.92833756 0.88953215]
    0.9179365787394577


    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)



```python
# Overfit
L.score(X_test, y_test)
```




    0.6759518356006168




```python
predictions = cross_val_predict(L, X, y, cv=kf)
plt.scatter(y, predictions)
accuracy = r2_score(y, predictions)
```

    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)



![IOWA]({{"/assets/img/P2_plot.png"}})



```python
# Looks like my model is overfit
# I ran this a few times, and didn't get nearly as bad of number. Looks like there was one fold that my model was just completely ill-fitting on.
scores = cross_val_score(L, X_test, y_test, cv=kf)
print(scores)
print(scores.mean())
```

    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    [ 0.76061406 -1.83481449  0.78095611  0.77575566  0.77518242]
    0.25153875067533676


# Code for submission to Kaggle


```python
# Clean up and make changes along the same lines to holdout_set
holdout_set.fillna(value=0, inplace=True)
```


```python
holdout_set.head(10)
```


```python
X_holdout = holdout_set[features]
```


```python
X_holdout.head(10)
```


```python
X_holdout.isnull().sum()
```


```python
# Replace one NA in Mas VNR Area
X_holdout.fillna(value=0, inplace=True)
```


```python
# Original linreg
y_preds = linreg.predict(X_holdout)
```


```python
# Lasso run:
y_preds = L.predict(X_holdout)
```


```python
y_preds
```


```python
my_ids = holdout_set['Id']
```


```python
df = pd.DataFrame()
```


```python
df['Id'] = my_ids
```


```python
df.set_index('Id', inplace=True)
```


```python
df['SalePrice'] = y_preds
```


```python
%pwd
```


```python
df.to_csv('./data/my_preds.csv')
```

# Graveyard - What is dead, may never die.


```python
# Was messing around trying to find a loop that could get me all numeric columns:
# .select_dtypes solved this issue
list_house_intFloat = []
for column in dfhouses.columns:
    if dfhouses[column].dtypes == int:
#         print('is int')
        list_house_intFloat.append(dfhouses[column])
#     elif dfhouses[column].dtypes == float:
#         dfhouse_int.append(dfhouses.loc[dfhouses[column]]
```


```python
# Counts for categorization of Street or Neighborhood
houses['Street'].value_counts()
```


```python
# Drop na's - was using before I did it on the original set
# X.dropna(axis=0, how='any', inplace=True)
```


```python
# houses_numonly_dropna = houses_numonly.dropna(axis=0, how='any')
# houses_numonly_dropna.isnull().sum()
```
