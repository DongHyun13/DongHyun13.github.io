---
layout: single
title:  "2019250035 이동현 6차 과제"
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


<a class="anchor" id="0"></a>

# **파이썬으로 구현하는 로지스틱 회귀 분류기**




이 튜토리얼에서는 파이썬과 Scikit-Learn을 사용하여 로지스틱 회귀(Logistic Regression)를 구현합니다. 호주 내일 비가 올지 여부를 예측하는 로지스틱 회귀 분류기를 구축하고, 로지스틱 회귀를 사용하여 이진 분류 모델을 학습합니다. 




<a class="anchor" id="0.1"></a>

# **목차**





1.	[로지스틱 회귀 소개](#1)

2.	[로지스틱 회귀 이해](#2)

3.	[로지스틱 회귀 가정](#3)

4.	[로지스틱 회귀 유형](#4)

5.	[라이브러리 Import](#5)

6.	[데이터셋 Import](#6)

7.	[탐색적 데이터 분석](#7)

8.	[피처 벡터 및 타겟 변수 선언](#8)

9.	[데이터를 학습 및 테스트 세트로 분리](#9)

10.	[피처 엔지니어링](#10)

11.	[피처 스케일링](#11)

12.	[모델 학습](#12)

13.	[결과 예측](#13)

14.	[정확도 점수 확인](#14)

15.	[혼동 행렬](#15)

16.	[분류 메트릭스](#16)

17.	[임계값 조정](#17)

18.	[ROC - AUC](#18)

19.	[k-Fold 교차 검증](#19)

20.	[그리드 서치를 사용한 하이퍼파라미터 최적화](#20)

21.	[결과 및 결론](#21)

22. [참고 문헌](#22)



# **1. 로지스틱 회귀 소개** <a class="anchor" id="1"></a>





[Table of Contents](#0.1)





새로운 분류 문제를 다룰 때, 데이터 과학자들이 먼저 떠오르는 알고리즘은 **로지스틱 회귀**일 것입니다. **로지스틱 회귀**는 이산적인 클래스로 예측을 하기 위해 사용되는 지도 학습 분류 알고리즘입니다. 실제로, 관측치를 서로 다른 범주로 분류하는 데 사용됩니다. 따라서, 출력 값은 이산적인 특징을 가지게 됩니다. **로지스틱 회귀**는 **로짓 회귀**(Logit Regression)라고도 불리며, 분류 문제를 해결하기 위해 가장 간단하고 직관적이며 다목적인 분류 알고리즘 중 하나입니다.


# **2. 로지스틱 회귀 이해** <a class="anchor" id="2"></a>





[Table of Contents](#0.1)





로지스틱 회귀 모델은 주로 분류 목적으로 사용되는 통계 모델입니다. 이는 주어진 관측치를 두 개 이상의 이산적인 클래스로 분류하는 데에 사용됩니다. 따라서 대상 변수는 이산적인 특성을 가집니다.

로지스틱 회귀 알고리즘은 다음과 같이 작동합니다.




## **선형 방정식 구현**





로지스틱 회귀 알고리즘은 독립 또는 설명 변수와 함께 선형 방정식을 구현하여 응답 값을 예측합니다. 예를 들어, 공부한 시간과 시험 통과 확률의 예를 고려해 보겠습니다. 여기서 공부한 시간은 설명 변수이며 x1로 표시됩니다. 시험 통과 확률은 반응 또는 대상 변수이며 z로 표시됩니다.





만약 설명 변수(x1)와 대상 변수(z)가 하나씩이라면, 선형 방정식은 다음과 같이 수학적으로 주어집니다.



    z = β0 + β1x1    



여기서 계수 β0과 β1은 모델의 매개 변수입니다.





만약 여러 개의 설명 변수가 있다면, 위의 방정식은 다음과 같이 확장될 수 있습니다.



    z = β0 + β1x1+ β2x2+……..+ βnxn

    

여기서 계수 β0, β1, β2 및 βn은 모델의 매개 변수입니다.




그러므로 예측된 응답 값은 위의 방정식으로 주어지며 z로 표시됩니다.



## **Sigmoid 함수**



로지스틱 회귀 모델에서 예측된 응답 값은 z로 나타내고, 이를 0과 1 사이의 확률 값으로 변환하는 데 시그모이드 함수를 사용합니다. 이 시그모이드 함수는 임의의 실수 값을 0과 1 사이의 확률 값으로 매핑합니다. 


머신러닝에서는 이 함수를 예측 값을 확률 값으로 매핑하기 위해 사용합니다. 시그모이드 함수는 S자 형태의 곡선을 가지며 시그모이드 곡선이라고도 합니다.


시그모이드 함수는 로지스틱 함수의 특수한 경우로 다음 수식으로 표현됩니다.



시각적으로는 다음 그래프로 시그모이드 함수를 나타낼 수 있습니다.

### Sigmoid 함수



![Sigmoid Function](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)


## **결정 경계**



시그모이드 함수는 0과 1 사이의 확률 값을 반환합니다. 이 확률 값을 이진 분류(합격/불합격, 예/아니오, 참/거짓)로 매핑하려면 임계값을 선택합니다. 이 임계값을 결정 경계라고 합니다. 결정 경계 이상의 값은 클래스 1로 매핑하고 이하의 값은 클래스 0으로 매핑합니다.



수학적으로는 다음과 같이 표현할 수 있습니다.



p ≥ 0.5 => class = 1



p < 0.5 => class = 0 



일반적으로 결정 경계는 0.5로 설정됩니다. 예를 들어, 확률 값이 0.8(> 0.5)인 경우, 해당 관측치를 클래스 1로 매핑합니다. 비슷하게, 확률 값이 0.2(< 0.5)인 경우, 해당 관측치를 클래스 0으로 매핑합니다. 이는 아래 그래프로 나타낼 수 있습니다.


![Decision boundary in sigmoid function](https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_sigmoid_w_threshold.png)


## **예측하기**



이제 우리는 로지스틱 회귀에서 시그모이드 함수와 결정 경계에 대해 알고 있습니다. 시그모이드 함수와 결정 경계에 대한 지식을 사용하여 예측 함수를 작성할 수 있습니다. 로지스틱 회귀의 예측 함수는 관측치가 긍정적인 (Yes 또는 True) P(class=1)에 속할 확률을 반환합니다. 확률이 1에 가까워질수록 모델이 관측치가 클래스 1에 속한다는 것이 확실해 집니다.
그렇지 않으면 클래스 0에 속한다고 합니다.



# **3. 로지스틱 회귀 가정** <a class="anchor" id="3"></a>





[Table of Contents](#0.1)





로지스틱 회귀 모델은 몇 가지 중요한 가정을 필요로 합니다. 이러한 가정은 다음과 같습니다.



1. 로지스틱 회귀 모델은 종속 변수가 이항, 다항 또는 순서형이어야 합니다.



2. 각 관측치는 서로 독립적이어야 합니다. 따라서 관측치는 반복 측정에서 오지 않아야 합니다.


3. 로지스틱 회귀 알고리즘은 독립 변수 간 다중공선성이 없거나 적어야 합니다. 이것은 독립 변수들이 서로 과도하게 상관 관계가 없어야 한다는 것을 의미합니다.


4. 로지스틱 회귀 모델은 독립 변수와 로그 오즈의 선형성을 가정합니다.


5. 로지스틱 회귀 모델의 성공은 샘플 크기에 달려 있습니다. 일반적으로 높은 정확도를 달성하려면 큰 샘플 크기가 필요합니다.



# **4. 로지스틱 회귀 유형** <a class="anchor" id="4"></a>





[Table of Contents](#0.1)





로지스틱 회귀 모델은 대상 변수 카테고리에 따라 세 가지 그룹으로 분류될 수 있습니다. 이 세 가지 그룹은 다음과 같습니다.


### 1. 바이너리 로지스틱 회귀


바이너리 로지스틱 회귀에서 대상 변수는 두 가지 가능한 카테고리를 가지고 있습니다. 대표적인 예로는 예 또는 아니오, 좋음 또는 나쁨, 참 또는 거짓, 스팸 또는 스팸이 아님, 통과 또는 실패 등이 있습니다.





### 2. 다항 로지스틱 회귀



다항 로지스틱 회귀에서 대상 변수는 서로 특정한 순서가 없는 세 개 이상의 카테고리가 있습니다. 따라서 이러한 경우 명목적인 카테고리가 세 개 이상입니다. 예를 들어, 과일 카테고리의 종류는 사과, 망고, 오렌지, 바나나 등이 있습니다.




### 3. 순차적 로지스틱 회귀


순차적 로지스틱 회귀에서 대상 변수는 세 개 이상의 순서형 카테고리를 가지고 있습니다. 따라서 이러한 카테고리에는 내재적인 순서가 포함됩니다. 예를 들어, 학생 성적은 낮음, 평균, 우수, 최우수와 같이 분류될 수 있습니다.



# **5. 라이브러리 임포트** <a class="anchor" id="5"></a>





[Table of Contents](#0.1)



```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

<pre>
/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv
</pre>

```python
import warnings

warnings.filterwarnings('ignore')
```

# **6. 데이터셋 임포트** <a class="anchor" id="6"></a>





[Table of Contents](#0.1)



```python
data = '/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv'

df = pd.read_csv(data)
```

# **7. 탐색적 데이터 분석** <a class="anchor" id="7"></a>





[Table of Contents](#0.1)





이제 데이터를 탐색하여 데이터에 대한 인사이트를 얻어보겠습니다.


```python
# view dimensions of dataset

df.shape
```

<pre>
(142193, 24)
</pre>
데이터셋에는 142193개의 인스턴스와 24개의 변수가 있습니다.


```python
# preview the dataset

df.head()
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
      <th>Date</th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>...</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RISK_MM</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>...</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>...</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>...</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>...</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>1.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>...</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>0.2</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



```python
col_names = df.columns

col_names
```

<pre>
Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow'],
      dtype='object')
</pre>
### RISK_MM 변수 삭제



데이터셋 설명에 따르면, RISK_MM 변수를 삭제해야 합니다. 따라서 다음과 같이 삭제해야 합니다.


```python
df.drop(['RISK_MM'], axis=1, inplace=True)
```


```python
# view summary of dataset

df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 142193 entries, 0 to 142192
Data columns (total 23 columns):
Date             142193 non-null object
Location         142193 non-null object
MinTemp          141556 non-null float64
MaxTemp          141871 non-null float64
Rainfall         140787 non-null float64
Evaporation      81350 non-null float64
Sunshine         74377 non-null float64
WindGustDir      132863 non-null object
WindGustSpeed    132923 non-null float64
WindDir9am       132180 non-null object
WindDir3pm       138415 non-null object
WindSpeed9am     140845 non-null float64
WindSpeed3pm     139563 non-null float64
Humidity9am      140419 non-null float64
Humidity3pm      138583 non-null float64
Pressure9am      128179 non-null float64
Pressure3pm      128212 non-null float64
Cloud9am         88536 non-null float64
Cloud3pm         85099 non-null float64
Temp9am          141289 non-null float64
Temp3pm          139467 non-null float64
RainToday        140787 non-null object
RainTomorrow     142193 non-null object
dtypes: float64(16), object(7)
memory usage: 25.0+ MB
</pre>
### 변수 유형





이번 섹션에서는 데이터셋을 범주형 변수와 수치형 변수로 분리하겠습니다. 데이터셋에는 범주형 변수와 수치형 변수가 혼합되어 있습니다. 범주형 변수는 데이터 유형이 객체입니다. 수치형 변수는 데이터 유형이 float64입니다.





먼저, 범주형 변수를 찾겠습니다.


```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

<pre>
There are 7 categorical variables

The categorical variables are : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
</pre>

```python
# view the categorical variables

df[categorical].head()
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
      <th>Date</th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>W</td>
      <td>W</td>
      <td>WNW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>WNW</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>WSW</td>
      <td>W</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>NE</td>
      <td>SE</td>
      <td>E</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>W</td>
      <td>ENE</td>
      <td>NW</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>


### 범주형 변수 요약




- 데이터셋에는 'Date' 열로 표시되는 날짜 변수가 있다.





-  'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday' 및 'RainTomorrow'로 구성된 6개의 범주형 변수가 있다.




- 'RainToday' 및 'RainTomorrow'은 바이너리 범주형 변수 중 두 개입니다.




- 'RainTomorrow'은 타겟 변수입니다.



## 범주형 변수 내의 문제 탐색





먼저 범주형 변수를 탐색해보겠습니다.




### 범주형 변수에서 결측값 찾기



```python
# check missing values in categorical variables

df[categorical].isnull().sum()
```

<pre>
Date                0
Location            0
WindGustDir      9330
WindDir9am      10013
WindDir3pm       3778
RainToday        1406
RainTomorrow        0
dtype: int64
</pre>

```python
# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
```

<pre>
WindGustDir     9330
WindDir9am     10013
WindDir3pm      3778
RainToday       1406
dtype: int64
</pre>
데이터셋에서 결측값이 있는 범주형 변수는 'WindGustDir', 'WindDir9am', 'WindDir3pm' 및 'RainToday'이며, 총 4개입니다.


### 범주형 변수의 빈도수




이제 범주형 변수의 빈도수를 확인해보겠습니다.



```python
# view frequency of categorical variables

for var in categorical: 
    
    print(df[var].value_counts())
```

<pre>
2014-04-15    49
2013-08-04    49
2014-03-18    49
2014-07-08    49
2014-02-27    49
              ..
2007-11-01     1
2007-12-30     1
2007-12-12     1
2008-01-20     1
2007-12-05     1
Name: Date, Length: 3436, dtype: int64
Canberra            3418
Sydney              3337
Perth               3193
Darwin              3192
Hobart              3188
Brisbane            3161
Adelaide            3090
Bendigo             3034
Townsville          3033
AliceSprings        3031
MountGambier        3030
Ballarat            3028
Launceston          3028
Albany              3016
Albury              3011
PerthAirport        3009
MelbourneAirport    3009
Mildura             3007
SydneyAirport       3005
Nuriootpa           3002
Sale                3000
Watsonia            2999
Tuggeranong         2998
Portland            2996
Woomera             2990
Cobar               2988
Cairns              2988
Wollongong          2983
GoldCoast           2980
WaggaWagga          2976
NorfolkIsland       2964
Penrith             2964
SalmonGums          2955
Newcastle           2955
CoffsHarbour        2953
Witchcliffe         2952
Richmond            2951
Dartmoor            2943
NorahHead           2929
BadgerysCreek       2928
MountGinini         2907
Moree               2854
Walpole             2819
PearceRAAF          2762
Williamtown         2553
Melbourne           2435
Nhil                1569
Katherine           1559
Uluru               1521
Name: Location, dtype: int64
W      9780
SE     9309
E      9071
N      9033
SSE    8993
S      8949
WSW    8901
SW     8797
SSW    8610
WNW    8066
NW     8003
ENE    7992
ESE    7305
NE     7060
NNW    6561
NNE    6433
Name: WindGustDir, dtype: int64
N      11393
SE      9162
E       9024
SSE     8966
NW      8552
S       8493
W       8260
SW      8237
NNE     7948
NNW     7840
ENE     7735
ESE     7558
NE      7527
SSW     7448
WNW     7194
WSW     6843
Name: WindDir9am, dtype: int64
SE     10663
W       9911
S       9598
WSW     9329
SW      9182
SSE     9142
N       8667
WNW     8656
NW      8468
ESE     8382
E       8342
NE      8164
SSW     8010
NNW     7733
ENE     7724
NNE     6444
Name: WindDir3pm, dtype: int64
No     109332
Yes     31455
Name: RainToday, dtype: int64
No     110316
Yes     31877
Name: RainTomorrow, dtype: int64
</pre>

```python
# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))
```

<pre>
2014-04-15    0.000345
2013-08-04    0.000345
2014-03-18    0.000345
2014-07-08    0.000345
2014-02-27    0.000345
                ...   
2007-11-01    0.000007
2007-12-30    0.000007
2007-12-12    0.000007
2008-01-20    0.000007
2007-12-05    0.000007
Name: Date, Length: 3436, dtype: float64
Canberra            0.024038
Sydney              0.023468
Perth               0.022455
Darwin              0.022448
Hobart              0.022420
Brisbane            0.022230
Adelaide            0.021731
Bendigo             0.021337
Townsville          0.021330
AliceSprings        0.021316
MountGambier        0.021309
Ballarat            0.021295
Launceston          0.021295
Albany              0.021211
Albury              0.021175
PerthAirport        0.021161
MelbourneAirport    0.021161
Mildura             0.021147
SydneyAirport       0.021133
Nuriootpa           0.021112
Sale                0.021098
Watsonia            0.021091
Tuggeranong         0.021084
Portland            0.021070
Woomera             0.021028
Cobar               0.021014
Cairns              0.021014
Wollongong          0.020979
GoldCoast           0.020957
WaggaWagga          0.020929
NorfolkIsland       0.020845
Penrith             0.020845
SalmonGums          0.020782
Newcastle           0.020782
CoffsHarbour        0.020768
Witchcliffe         0.020761
Richmond            0.020753
Dartmoor            0.020697
NorahHead           0.020599
BadgerysCreek       0.020592
MountGinini         0.020444
Moree               0.020071
Walpole             0.019825
PearceRAAF          0.019424
Williamtown         0.017954
Melbourne           0.017125
Nhil                0.011034
Katherine           0.010964
Uluru               0.010697
Name: Location, dtype: float64
W      0.068780
SE     0.065467
E      0.063794
N      0.063526
SSE    0.063245
S      0.062936
WSW    0.062598
SW     0.061867
SSW    0.060552
WNW    0.056726
NW     0.056283
ENE    0.056205
ESE    0.051374
NE     0.049651
NNW    0.046142
NNE    0.045241
Name: WindGustDir, dtype: float64
N      0.080123
SE     0.064434
E      0.063463
SSE    0.063055
NW     0.060144
S      0.059729
W      0.058090
SW     0.057928
NNE    0.055896
NNW    0.055136
ENE    0.054398
ESE    0.053153
NE     0.052935
SSW    0.052380
WNW    0.050593
WSW    0.048125
Name: WindDir9am, dtype: float64
SE     0.074990
W      0.069701
S      0.067500
WSW    0.065608
SW     0.064574
SSE    0.064293
N      0.060952
WNW    0.060875
NW     0.059553
ESE    0.058948
E      0.058667
NE     0.057415
SSW    0.056332
NNW    0.054384
ENE    0.054321
NNE    0.045319
Name: WindDir3pm, dtype: float64
No     0.768899
Yes    0.221213
Name: RainToday, dtype: float64
No     0.775819
Yes    0.224181
Name: RainTomorrow, dtype: float64
</pre>
### 레이블 수: cardinality





범주형 변수 내의 레이블 수를 카디널리티(cardinality)라고 합니다. 변수 내 레이블의 수가 많을수록 고카디널리티(high cardinality)입니다. 고카디널리티는 머신 러닝 모델에서 일부 심각한 문제를 야기할 수 있으므로, 고카디널리티를 확인해보겠습니다.



```python
# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
```

<pre>
Date  contains  3436  labels
Location  contains  49  labels
WindGustDir  contains  17  labels
WindDir9am  contains  17  labels
WindDir3pm  contains  17  labels
RainToday  contains  3  labels
RainTomorrow  contains  2  labels
</pre>
날짜 변수가 존재하며 전처리가 필요한 것으로 나타납니다. 다음 섹션에서 전처리를 진행하겠습니다.





다른 변수들은 상대적으로 작은 변수 수를 갖고 있습니다.


### 날짜 변수의 피처 엔지니어링


```python
df['Date'].dtypes
```

<pre>
dtype('O')
</pre>
데이터셋에서 Date 변수의 데이터 타입은 object입니다. 따라서 Date 변수를 datetime 형식으로 파싱할 것입니다.



```python
# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])
```


```python
# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()
```

<pre>
0    2008
1    2008
2    2008
3    2008
4    2008
Name: Year, dtype: int64
</pre>

```python
# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head()
```

<pre>
0    12
1    12
2    12
3    12
4    12
Name: Month, dtype: int64
</pre>

```python
# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()
```

<pre>
0    1
1    2
2    3
3    4
4    5
Name: Day, dtype: int64
</pre>

```python
# again view the summary of dataset

df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 142193 entries, 0 to 142192
Data columns (total 26 columns):
Date             142193 non-null datetime64[ns]
Location         142193 non-null object
MinTemp          141556 non-null float64
MaxTemp          141871 non-null float64
Rainfall         140787 non-null float64
Evaporation      81350 non-null float64
Sunshine         74377 non-null float64
WindGustDir      132863 non-null object
WindGustSpeed    132923 non-null float64
WindDir9am       132180 non-null object
WindDir3pm       138415 non-null object
WindSpeed9am     140845 non-null float64
WindSpeed3pm     139563 non-null float64
Humidity9am      140419 non-null float64
Humidity3pm      138583 non-null float64
Pressure9am      128179 non-null float64
Pressure3pm      128212 non-null float64
Cloud9am         88536 non-null float64
Cloud3pm         85099 non-null float64
Temp9am          141289 non-null float64
Temp3pm          139467 non-null float64
RainToday        140787 non-null object
RainTomorrow     142193 non-null object
Year             142193 non-null int64
Month            142193 non-null int64
Day              142193 non-null int64
dtypes: datetime64[ns](1), float64(16), int64(3), object(6)
memory usage: 28.2+ MB
</pre>
Date 변수에서 파생된 새로운 3개의 열이 생성된 것을 확인할 수 있습니다. 이제 원래의 'Date' 변수를 데이터셋에서 삭제할 것입니다.



```python
# drop the original Date variable

df.drop('Date', axis=1, inplace = True)
```


```python
# preview the dataset again

df.head()
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
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>WNW</td>
      <td>...</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>...</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>WSW</td>
      <td>...</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>E</td>
      <td>...</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>NW</td>
      <td>...</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>


데이터셋에서 'Date' 변수가 삭제된 것을 확인할 수 있습니다.



### 범주형 변수 조사하기





이제 각각의 범주형 변수를 하나씩 조사해보겠습니다.



```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

<pre>
There are 6 categorical variables

The categorical variables are : ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
</pre>
데이터셋에는 총 6개의 범주형 변수가 존재합니다. 'Date' 변수는 삭제되었습니다. 먼저, 범주형 변수에서 결측값을 확인해보겠습니다.


```python
# check for missing values in categorical variables 

df[categorical].isnull().sum()
```

<pre>
Location            0
WindGustDir      9330
WindDir9am      10013
WindDir3pm       3778
RainToday        1406
RainTomorrow        0
dtype: int64
</pre>
`WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` 변수에 결측값이 존재하는 것을 확인할 수 있습니다. 이제 각각의 변수를 조사해보겠습니다.


### `Location` 변수 탐색



```python
# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')
```

<pre>
Location contains 49 labels
</pre>

```python
# check labels in location variable

df.Location.unique()
```

<pre>
array(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru'], dtype=object)
</pre>

```python
# check frequency distribution of values in Location variable

df.Location.value_counts()
```

<pre>
Canberra            3418
Sydney              3337
Perth               3193
Darwin              3192
Hobart              3188
Brisbane            3161
Adelaide            3090
Bendigo             3034
Townsville          3033
AliceSprings        3031
MountGambier        3030
Ballarat            3028
Launceston          3028
Albany              3016
Albury              3011
PerthAirport        3009
MelbourneAirport    3009
Mildura             3007
SydneyAirport       3005
Nuriootpa           3002
Sale                3000
Watsonia            2999
Tuggeranong         2998
Portland            2996
Woomera             2990
Cobar               2988
Cairns              2988
Wollongong          2983
GoldCoast           2980
WaggaWagga          2976
NorfolkIsland       2964
Penrith             2964
SalmonGums          2955
Newcastle           2955
CoffsHarbour        2953
Witchcliffe         2952
Richmond            2951
Dartmoor            2943
NorahHead           2929
BadgerysCreek       2928
MountGinini         2907
Moree               2854
Walpole             2819
PearceRAAF          2762
Williamtown         2553
Melbourne           2435
Nhil                1569
Katherine           1559
Uluru               1521
Name: Location, dtype: int64
</pre>

```python
# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()
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
      <th>Albany</th>
      <th>Albury</th>
      <th>AliceSprings</th>
      <th>BadgerysCreek</th>
      <th>Ballarat</th>
      <th>Bendigo</th>
      <th>Brisbane</th>
      <th>Cairns</th>
      <th>Canberra</th>
      <th>Cobar</th>
      <th>...</th>
      <th>Townsville</th>
      <th>Tuggeranong</th>
      <th>Uluru</th>
      <th>WaggaWagga</th>
      <th>Walpole</th>
      <th>Watsonia</th>
      <th>Williamtown</th>
      <th>Witchcliffe</th>
      <th>Wollongong</th>
      <th>Woomera</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>


### `WindGustDir` 변수 



```python
# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
```

<pre>
WindGustDir contains 17 labels
</pre>

```python
# check labels in WindGustDir variable

df['WindGustDir'].unique()
```

<pre>
array(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE',
       'S', 'NW', 'SE', 'ESE', nan, 'E', 'SSW'], dtype=object)
</pre>

```python
# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()
```

<pre>
W      9780
SE     9309
E      9071
N      9033
SSE    8993
S      8949
WSW    8901
SW     8797
SSW    8610
WNW    8066
NW     8003
ENE    7992
ESE    7305
NE     7060
NNW    6561
NNE    6433
Name: WindGustDir, dtype: int64
</pre>

```python
# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
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
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE    7992
ESE    7305
N      9033
NE     7060
NNE    6433
NNW    6561
NW     8003
S      8949
SE     9309
SSE    8993
SSW    8610
SW     8797
W      9780
WNW    8066
WSW    8901
NaN    9330
dtype: int64
</pre>
'WindGustDir' 변수에 결측값이 9330개 있는 것을 확인했습니다.


### `WindDir9am` 변수 



```python
# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
```

<pre>
WindDir9am contains 17 labels
</pre>

```python
# check labels in WindDir9am variable

df['WindDir9am'].unique()
```

<pre>
array(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', nan, 'SSW', 'N',
       'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], dtype=object)
</pre>

```python
# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()
```

<pre>
N      11393
SE      9162
E       9024
SSE     8966
NW      8552
S       8493
W       8260
SW      8237
NNE     7948
NNW     7840
ENE     7735
ESE     7558
NE      7527
SSW     7448
WNW     7194
WSW     6843
Name: WindDir9am, dtype: int64
</pre>

```python
# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
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
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE     7735
ESE     7558
N      11393
NE      7527
NNE     7948
NNW     7840
NW      8552
S       8493
SE      9162
SSE     8966
SSW     7448
SW      8237
W       8260
WNW     7194
WSW     6843
NaN    10013
dtype: int64
</pre>
`WindDir9am` 변수에 결측값이 10013개 있는 것을 확인했습니다.

### `WindDir3pm` 변수 탐색



```python
# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
```

<pre>
WindDir3pm contains 17 labels
</pre>

```python
# check labels in WindDir3pm variable

df['WindDir3pm'].unique()
```

<pre>
array(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
       'SW', 'SE', 'N', 'S', 'NNE', nan, 'NE'], dtype=object)
</pre>

```python
# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()
```

<pre>
SE     10663
W       9911
S       9598
WSW     9329
SW      9182
SSE     9142
N       8667
WNW     8656
NW      8468
ESE     8382
E       8342
NE      8164
SSW     8010
NNW     7733
ENE     7724
NNE     6444
Name: WindDir3pm, dtype: int64
</pre>

```python
# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
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
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE     7724
ESE     8382
N       8667
NE      8164
NNE     6444
NNW     7733
NW      8468
S       9598
SE     10663
SSE     9142
SSW     8010
SW      9182
W       9911
WNW     8656
WSW     9329
NaN     3778
dtype: int64
</pre>
`WindDir3pm` 변수에 결측값이 3778개 있는 것을 확인했습니다.

### `RainToday` 변수 확인



```python
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
```

<pre>
RainToday contains 3 labels
</pre>

```python
# check labels in WindGustDir variable

df['RainToday'].unique()
```

<pre>
array(['No', 'Yes', nan], dtype=object)
</pre>

```python
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()
```

<pre>
No     109332
Yes     31455
Name: RainToday, dtype: int64
</pre>

```python
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
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
      <th>Yes</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
Yes    31455
NaN     1406
dtype: int64
</pre>
`RainToday` 변수에 결측값이 1406개 있는 것을 확인했습니다.

### 수치형 변수에 대한 탐색



```python
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
```

<pre>
There are 19 numerical variables

The numerical variables are : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']
</pre>

```python
# view the numerical variables

df[numerical].head()
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>20.0</td>
      <td>24.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>46.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


### 수치형 변수에 대한 탐색 결과 요약





- 총 16개의 수치형 변수가 있습니다.





- 이는 `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am` and `Temp3pm` 입니다.





- 모든 수치형 변수는 연속형입니다.


## 수치형 변수에서 발견된 문제점 탐색





이제 수치형 변수에 대해 탐색해보겠습니다.





### 수치형 변수에서 누락된 값



```python
# check missing values in numerical variables

df[numerical].isnull().sum()
```

<pre>
MinTemp            637
MaxTemp            322
Rainfall          1406
Evaporation      60843
Sunshine         67816
WindGustSpeed     9270
WindSpeed9am      1348
WindSpeed3pm      2630
Humidity9am       1774
Humidity3pm       3610
Pressure9am      14014
Pressure3pm      13981
Cloud9am         53657
Cloud3pm         57094
Temp9am            904
Temp3pm           2726
Year                 0
Month                0
Day                  0
dtype: int64
</pre>
16개의 수치형 변수 모두 누락된 값이 포함되어 있음을 확인할 수 있습니다.


### 수치형 변수에서 이상치



```python
# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)
```

<pre>
        MinTemp   MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
count  141556.0  141871.0  140787.0      81350.0   74377.0       132923.0   
mean       12.0      23.0       2.0          5.0       8.0           40.0   
std         6.0       7.0       8.0          4.0       4.0           14.0   
min        -8.0      -5.0       0.0          0.0       0.0            6.0   
25%         8.0      18.0       0.0          3.0       5.0           31.0   
50%        12.0      23.0       0.0          5.0       8.0           39.0   
75%        17.0      28.0       1.0          7.0      11.0           48.0   
max        34.0      48.0     371.0        145.0      14.0          135.0   

       WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  Pressure9am  \
count      140845.0      139563.0     140419.0     138583.0     128179.0   
mean           14.0          19.0         69.0         51.0       1018.0   
std             9.0           9.0         19.0         21.0          7.0   
min             0.0           0.0          0.0          0.0        980.0   
25%             7.0          13.0         57.0         37.0       1013.0   
50%            13.0          19.0         70.0         52.0       1018.0   
75%            19.0          24.0         83.0         66.0       1022.0   
max           130.0          87.0        100.0        100.0       1041.0   

       Pressure3pm  Cloud9am  Cloud3pm   Temp9am   Temp3pm      Year  \
count     128212.0   88536.0   85099.0  141289.0  139467.0  142193.0   
mean        1015.0       4.0       5.0      17.0      22.0    2013.0   
std            7.0       3.0       3.0       6.0       7.0       3.0   
min          977.0       0.0       0.0      -7.0      -5.0    2007.0   
25%         1010.0       1.0       2.0      12.0      17.0    2011.0   
50%         1015.0       5.0       5.0      17.0      21.0    2013.0   
75%         1020.0       7.0       7.0      22.0      26.0    2015.0   
max         1040.0       9.0       9.0      40.0      47.0    2017.0   

          Month       Day  
count  142193.0  142193.0  
mean        6.0      16.0  
std         3.0       9.0  
min         1.0       1.0  
25%         3.0       8.0  
50%         6.0      16.0  
75%         9.0      23.0  
max        12.0      31.0   2
</pre>
자세히 살펴보면, `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` 열에는 이상치가 포함되어 있을 수 있습니다.






상기 변수에서 이상치를 시각화하기 위해 상자 그림을 그려보겠습니다.



```python
# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```

<pre>
Text(0, 0.5, 'WindSpeed3pm')
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA34AAAJCCAYAAACf5hV2AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzs3XuU3XV97//nO7eJDWBANCIgwSO1E8cqNkc9MOesGaNVrArWG8HjjWnSiIxtqRJg1qpaz1QpFQ/GCiYdBP3pIGjF4BUaZ+pvStEGvGFGgR/X4RJQieZCbpP374/9nTATJmFIsue7957nY6299v5+9nd/90tXZn1478/lG5mJJEmSJKlxTSs7gCRJkiSpuiz8JEmSJKnBWfhJkiRJUoOz8JMkSZKkBmfhJ0mSJEkNzsJPkiRJkhqchZ8kSZIkNTgLP0mSJElqcBZ+kiRJktTgZpQd4EAceeSROX/+/LJjSJNq8+bNzJkzp+wY0qS7+eabf52Zzyw7R72wj9RUZB+pqWii/WNdF37z589n7dq1ZceQJlV/fz9tbW1lx5AmXUTcU3aGemIfqanIPlJT0UT7R6d6SpIkSVKDs/CTJEmSpAZn4SdJkiRJDc7CT5IkSZIanIWfJEmSJDU4Cz9JkiRJanAWflKd6O3tpaWlhUWLFtHS0kJvb2/ZkSRJqgn2kdKTq+v7+ElTRW9vL11dXfT09DA8PMz06dPp6OgAYPHixSWnkySpPPaR0sQ44ifVge7ubs444ww6Ozt5zWteQ2dnJ2eccQbd3d1lR5MkqVTd3d309PTQ3t7OjBkzaG9vp6enxz5S2oMjflIdWLduHVu2bHnCr5l333132dEkSSrV4OAgra2tY9paW1sZHBwsKZFUmxzxk+rArFmzOPvss8f8mnn22Wcza9assqNJklSq5uZmBgYGxrQNDAzQ3NxcUiKpNln4SXVg+/btrFixgr6+Pnbu3ElfXx8rVqxg+/btZUeTJKlUXV1ddHR0jOkjOzo66OrqKjuaVFOc6inVgQULFnDaaafR2dnJ4OAgzc3NvOMd7+Daa68tO5okSaUa2cBldB/Z3d3txi7SHiz8pDrQ1dU17o5lLlyXJKlS/C1evJj+/n7a2trKjiPVJAs/qQ74a6bUWCLicuD1wMOZ2bLHex8ELgKemZm/jogALgFeB2wB3pOZt0x2ZklSfXONn1QnFi9ezK233sqaNWu49dZbLfqk+nYF8No9GyPiWODVwL2jmk8BTigeS4FLJyGfJKnBWPhJkjTJMvMHwG/HeetTwLlAjmo7FfhCVtwEzI2IoyYhpiSpgTjVU5KkGhARbwTuz8yfVmZ37nY0cN+o46Gi7cFxrrGUyqgg8+bNo7+/v2p5pVq0adMm/91Le2HhJ0lSySLiD4Au4E/He3ucthynjcxcCawEWLhwYbrJhaYaN3eR9q5qUz0jYnZE/CgifhoRv4iIjxbtV0TEXRHxk+LxkqI9IuLTEXFHRPwsIl5arWySJNWY/wYcD/w0Iu4GjgFuiYhnUxnhO3bUuccAD0x6QklSXavmiN824JWZuSkiZgIDEfGd4r0PZeZX9zh/9OL1l1NZvP7yKuaTJKkmZObPgWeNHBfF38JiV8/VwNkRcRWVfvF3mfmEaZ6SJO1L1Ub8ikXom4rDmcVj3KkpBRevS5KmhIjoBf4TeEFEDEVExz5O/zZwJ3AHsAo4axIiSpIaTFXX+EXEdOBm4PnAP2fmDyPifUB3RPwdsAY4LzO3McHF6y5c11TnwnWp/mXmPu/HkpnzR71O4P3VziRJamxVLfwycxh4SUTMBb4eES3A+cBDwCwqC9CXA3/PBBevu3BdU50L1yVJkvRUTcp9/DJzA9APvDYzHyymc24DPg+8rDjNxeuSJEmSVAXV3NXzmcVIHxHxNOBVwC9H1u1F5SZFpwG3Fh9ZDbyr2N3zFbh4XZIkSZIOimpO9TwKuLJY5zcNuDozvxkR34+IZ1KZ2vkTYFlx/reB11FZvL4FeG8Vs0mSJEnSlFG1wi8zfwacOE77K/dyvovXJUmSJKkKJmWNnyRJkiSpPBZ+kiRJktTgLPwkSZIkqcFZ+EmSJElSg7PwkyRJkqQGZ+EnSZIkSQ3Owk+SJEmSGpyFnyRJkiQ1OAs/SZIkSWpwFn6SJEmS1OAs/CRJkiSpwVn4SZIkSVKDs/CTJEmSpAZn4SdJkiRJDc7CT5IkSZIanIWfJEmSJDU4Cz9JkiRJanAWfpIkTbKIuDwiHo6IW0e1XRQRv4yIn0XE1yNi7qj3zo+IOyLiVxHxmnJSS5LqmYWfJEmT7wrgtXu03QC0ZOYfA7cB5wNExALgdOCFxWc+GxHTJy+qJKkRWPhJkjTJMvMHwG/3aLs+M3cWhzcBxxSvTwWuysxtmXkXcAfwskkLK0lqCDPKDiBJkp7gTOArxeujqRSCI4aKtieIiKXAUoB58+bR399fxYhS7dm0aZP/7qW9qFrhFxGzgR8ATcX3fDUzPxwRxwNXAUcAtwDvzMztEdEEfAH4E+A3wNsz8+5q5ZMkqRZFRBewE/jSSNM4p+V4n83MlcBKgIULF2ZbW1s1Iko1q7+/H//dS+Or5lTPbcArM/PFwEuA10bEK4ALgU9l5gnAo0BHcX4H8GhmPh/4VHGeJElTRkS8G3g98I7MHCnuhoBjR512DPDAZGeTJNW3qhV+WbGpOJxZPBJ4JfDVov1K4LTi9anFMcX7iyJivF85JUlqOBHxWmA58MbM3DLqrdXA6RHRVMyaOQH4URkZJUn1q6pr/Ipdx24Gng/8M/D/ARtGLV4fvU7haOA+gMzcGRG/A54B/HqPa7p+QVOa6xek+hcRvUAbcGREDAEfprKLZxNwQ/G7502ZuSwzfxERVwPrqEwBfX9mDpeTXJJUr6pa+BUd00uKexF9HWge77TieUJrGFy/oKnO9QtS/cvMxeM09+zj/G6gu3qJJEmNblJu55CZG4B+4BXA3IgYKThHr1PYvYaheP/p7LHVtSRJkiTpqata4RcRzyxG+oiIpwGvAgaBPuAtxWnvBr5RvF5dHFO8//1RC9slSZIkSfupmlM9jwKuLNb5TQOuzsxvRsQ64KqI+D/Aj3l8aksP8MWIuIPKSN/pVcwmSZIkSVNG1Qq/zPwZcOI47XcCLxunfSvw1mrlkSRJkqSpalLW+EmSJEmSymPhJ0mSJEkNzsJPkiRJkhqchZ8kSZIkNTgLP6lO9Pb20tLSwqJFi2hpaaG3t7fsSJIkSaoT1bydg6SDpLe3l66uLnp6ehgeHmb69Ol0dHQAsHjx4pLTSZIkqdY54ifVge7ubnp6emhvb2fGjBm0t7fT09NDd3d32dEkSZJUByz8pDowODhIa2vrmLbW1lYGBwdLSiRJkqR6YuEn1YHm5mYGBgbGtA0MDNDc3FxSIkmSJNUTCz+pDnR1ddHR0UFfXx87d+6kr6+Pjo4Ourq6yo4mSZKkOuDmLlIdWLx4MTfeeCOnnHIK27Zto6mpiSVLlrixiyRJkibEwk+qA729vXzrW9/iO9/5zphdPU866SSLP0mSJD0pp3pKdcBdPSVJknQgLPykOuCunpIkSToQFn5SHXBXT0mSJB0ICz+pDrirpyRJkg6Em7tIdWBkA5fOzk4GBwdpbm6mu7vbjV0kSZI0IRZ+Up1YvHgxixcvpr+/n7a2trLjSJIkqY441VOSpEkWEZdHxMMRceuotiMi4oaIuL14Prxoj4j4dETcERE/i4iXlpdcklSvLPwkSZp8VwCv3aPtPGBNZp4ArCmOAU4BTigeS4FLJymjJKmBWPhJkjTJMvMHwG/3aD4VuLJ4fSVw2qj2L2TFTcDciDhqcpJKkhpF1db4RcSxwBeAZwO7gJWZeUlEfARYAjxSnHpBZn67+Mz5QAcwDHwgM79XrXySJNWYeZn5IEBmPhgRzyrajwbuG3XeUNH24J4XiIilVEYFmTdvHv39/VUNLNWaTZs2+e9e2otqbu6yE/jbzLwlIg4Fbo6IG4r3PpWZ/zT65IhYAJwOvBB4DvBvEfGHmTlcxYySJNW6GKctxzsxM1cCKwEWLlyYbgSlqcYN0KS9q9pUz8x8MDNvKV5vBAap/EK5N6cCV2Xmtsy8C7gDeFm18kn1pre3l5aWFhYtWkRLSwu9vb1lR5J0cK0fmcJZPD9ctA8Bx4467xjggUnOJkmqc5NyO4eImA+cCPwQOBk4OyLeBaylMir4KJWi8KZRHxuZyiJNeb29vXR1ddHT08Pw8DDTp0+no6MDwHv5SY1jNfBu4BPF8zdGtZ8dEVcBLwd+NzIlVJKkiap64RcRhwBfA/46M38fEZcCH6MyTeVjwCeBM5ngVBbXL2gquuCCC/jABz5ARLB161YOOeQQOjs7ueCCCzjqKPd4kOpNRPQCbcCRETEEfJhKwXd1RHQA9wJvLU7/NvA6KjNhtgDvnfTAkqS6F5njLhM4OBePmAl8E/heZl48zvvzgW9mZkuxsQuZ+fHive8BH8nM/9zb9RcuXJhr166tRnSppkyfPp2tW7cyc+bM3esXduzYwezZsxkedhmspoaIuDkzF5ado17YR2oqco2fpqKJ9o9VW+MXEQH0AIOji749tqB+EzBy89rVwOkR0RQRx1O5X9GPqpVPqifNzc0MDAyMaRsYGKC5ubmkRJIkSaon1ZzqeTLwTuDnEfGTou0CYHFEvITKNM67gb8EyMxfRMTVwDoqO4K+3x09pYquri7e/va3M2fOHO655x6OO+44Nm/ezCWXXFJ2NEmSJNWBqhV+mTnA+Ov2vr2Pz3QD3dXKJDWCymC6JEmSNHFVm+op6eDp7u5m6dKlzJkzB4A5c+awdOlSurv9nUSSJElPblJu5yDpwKxbt44tW7Y84XYOd999d9nRJEmSVAcc8ZPqwKxZszj77LNpb29nxowZtLe3c/bZZzNr1qyyo0mSJKkOOOIn1YHt27ezYsUKTjzxRIaHh+nr62PFihVs37697GiSJEmqAxZ+Uh1YsGABp512Gp2dnQwODtLc3Mw73vEOrr322rKjSZIkqQ5Y+El1oKuri66uries8XNzF0mSJE2EhZ9UBxYvXgwwZsSvu7t7d7skSZK0L27uIknSAYiIP4+I2yPidxHx+4jYGBG/LzuXJEmjOeIn1YHe3t5xp3oCjvpJ5ftH4A2ZOVh2EEmS9sYRP6kOdHd309PTM+Z2Dj09Pa7xk2rDeos+SVKts/CT6sDg4CBDQ0O0tLSwaNEiWlpaGBoaYnDQ/9aUasDaiPhKRCwupn3+eUT8edmhpKmkt7d3TB/Z29tbdiSp5jjVU6oDz3nOczj33HP58pe/vHuq5xlnnMFznvOcsqNJgsOALcCfjmpL4F/LiSNNLS6HkCbGwk+qE1u3buXMM8/knnvu4bjjjmPr1q0ccsghZceSprzMfG/ZGaSpbPRyiP7+ftra2ujp6aGzs9PCTxrFqZ5SHbj//vuZOXMmABEBwMyZM7n//vvLjCUJiIhjIuLrEfFwRKyPiK9FxDFl55KmisHBQVpbW8e0tba2uhxC2oOFn1QHZs2axXnnncddd93FmjVruOuuuzjvvPOYNWtW2dEkweeB1cBzgKOB64o2SZOgubmZgYGBMW0DAwM0NzeXlEiqTU71lOrA9u3bWbFiBSeeeCLDw8P09fWxYsUKtm/fXnY0SfDMzBxd6F0REX9dWhppiunq6uLtb387c+bM4d577+W5z30umzdv5pJLLik7mlRTLPykOrBgwQJOOOEETjnlFLZt20ZTUxOnnHIKc+bMKTuaJPh1RPxvYGQbwcXAb0rMI01ZmVl2BKlmOdVTqgPt7e2sXr2auXPnAjB37lxWr15Ne3t7yckkAWcCbwMeAh4E3lK0SZoE3d3dLF26lDlz5hARzJkzh6VLl3qvW2kP+xzxi4ifU9mS+glvAZmZf1yVVJLGuPbaa5k+fTrr168HYP369cycOZNrr72WFStWlJxOmtoy817gjWXnkKaqdevWsXnzZi6//PLdt3MY2QVb0uOebKrn6yclhaR9GhoaYtq0aXzyk59kwYIFrFu3jg996EMMDQ2VHU2asiLi3Mz8x4hYwTg/kmbmB/bzun8D/EVxzZ8D7wWOAq4CjgBuAd6ZmS7ylahsgNbZ2Tnmdg6dnZ1ccMEFZUeTaso+C7/M9KcSqUb8xV/8Beeccw79/f2cc845/OpXv2LlypVlx5KmspG94tcerAtGxNHAB4AFmflYRFwNnA68DvhUZl4VEZcBHcClB+t7pXq2fft2PvOZz4zZAO0zn/mMG6BJe3iyqZ4b2fdUz8P28dljgS8AzwZ2ASsz85KIOAL4CjAfuBt4W2Y+GpWbk11CpXPbArwnM295yv+LpAa1evVqTj/99N2d2urVq8uOJE1pmXld8XJLZl4z+r2IeOsBXHoG8LSI2AH8AZV1g68EzijevxL4CBZ+ElDZAO20006js7OTwcFBmpubOeOMM7j22mvLjibVlCcb8Tv0AK69E/jbzLwlIg4Fbo6IG4D3AGsy8xMRcR5wHrAcOAU4oXi8nEqH9vID+H6pYcyYMYONGzdy5pln7t6qeuPGjcyY4ca8Ug04H7hmAm1PKjPvj4h/Au4FHgOuB24GNmTmzuK0ISr3C3yCiFgKLAWYN28e/f39TzWCVHfe9KY30dPTw4c+9CGOP/547rrrLi666CI6Ojr8G5BGeUr/1RgRzwJmjxwXC9rHlZkPUvmVkszcGBGDVDqqU4G24rQrgX4qhd+pwBeysg/vTRExNyKOKq4jTWnLli3js5/9LI899hi7du3iscce47HHHuOss84qO5o0ZUXEKVRmqRwdEZ8e9dZhVH783J9rHk6lPzwe2ECleDxlnFPH3bM+M1cCKwEWLlyYbW1t+xNDqittbW1s2LCB888/f/ctj5YsWcLHPvaxsqNJNWVChV9EvBH4JPAc4GHgOCprG144wc/PB04EfgjMGynmMvPBopiESlF436iPjfyiaeGnKW9k585Vq1YBsGHDBs466yx39JTK9QCV9X1vpDIqN2Ij8Df7ec1XAXdl5iMAEfGvwEnA3IiYUYz6HVN8tySgt7eXb33rW3znO9/ZvatnR0cHJ510EosXLy47nlQzJjri9zHgFcC/ZeaJEdFO5Qa1TyoiDgG+Bvx1Zv6+spRv/FPHaXvCL5pOY9FU9eY3v5k3v/nNbNq0iUMOOQTAf/9SiTLzp8BPI+LLmbnjIF32XuAVEfEHVKZ6LqJSXPZRuT/gVcC7gW8cpO+T6l53dzc9PT1jdvXs6emhs7PTwk8aZaKF347M/E1ETIuIaZnZFxEXPtmHImImlaLvS5n5r0Xz+pEpnBFxFJURRKiM8B076uPj/qLpNBZNdSOdmqSaMT8iPg4sYOxyiOc91Qtl5g8j4qtUbtmwE/gxlT7vW8BVEfF/iraegxFcagSDg4O0traOaWttbWVwcHAvn5CmpmkTPG9DMXL3A+BLEXEJT7J+odilswcYzMyLR721msqvlTD2V8vVwLui4hXA71zfJz2ut7eXlpYWFi1aREtLC729vWVHklTxeSobku0E2qnsaP3F/b1YZn44M/8oM1sy852ZuS0z78zMl2Xm8zPzrZm57SBll+pec3MzH/3oR8f0kR/96Edpbm4uO5pUU57sdg5NRedyKrCVypqFdwBPB/7+Sa59MvBO4OcR8ZOi7QLgE8DVEdFBZUrLyJbX36aySP4OKrdzeO9T/l8jNaje3l66urro6ekZs34BcBqLVL6nZeaaiIji/rcfiYj/F/hw2cGkqaC9vZ0LL7yQCy+8kAULFrBu3TqWL1/OsmXLyo4m1ZSobKK5lzcjbsnMl0bEFzPznZOYa0IWLlyYa9cetPvmSjWrpaWF0047jWuvvXb3PYpGjm+99day40mTIiJuzsyFZefYU0T8B/A/ga8C3wfuBz6RmS8oM5d9pKYK+0hNdRPtH5+s8LsVuAj4O+BDe74/at1eKezUNFVMmzaN4447jssvv3z3iN+ZZ57JPffcw65du8qOJ02KGi78/juVna7nUtkM7TDgosy8qcxc9pGaKqZPn87WrVuZOXPm7nXwO3bsYPbs2QwPD5cdT6q6ifaPT7a5yzIqUzvnAm/Y470ESi38pKli1qxZnHzyyXR2du7+NfPkk0/mwQddBiuVKSKmA2/LzA8Bm3CZgjTpRtb47Tni5xo/aax9Fn6ZOQAMRMTazHQHMakk27Zt48tf/jLTpk1j165d/PKXv2TdunXsa8ReUvVl5nBE/Emxvs8/SKkErvGTJmZCt3PIzJ6IOAmYP/ozmfmFKuWSNMr06dMZHh7ePWVl5Hn69OllxpJU8WPgGxFxDbB5pLHs5RDSVNHX18fy5cu5/PLLd4/4LV++nGuvvbbsaFJNmdDtHCLii8A/Aa3Afy8eNbfOQmpUI4Xe+973Pq677jre9773jWmXVKojgN8Ar6SyLOINwOtLTSRNIYODg7zgBWP3UnrBC17gffykPexzc5fdJ0UMAgtqbRqLC9c1VUQEzc3N3HnnnWzbto2mpiae97znMTg46HRPTRm1urlLrbKP1FRx7LHHsmnTJubOncs999zDcccdx4YNGzjkkEO47777yo4nVd1E+8eJ3sD9VuDZBxZJ0oEYHBxk7ty5RARz5871l0ypRkTEMRHx9Yh4OCLWR8TXIuKYsnNJU8WWLVvYsGEDQ0NDZCZDQ0Ns2LCBLVu2lB1NqikTLfyOBNZFxPciYvXIo5rBJD3R+vXryUzWr19fdhRJj/s8sBp4DnA0cF3RJmkS/Pa3vyUieMYzngHAM57xDCKC3/72tyUnk2rLRAu/jwCnAf8AfHLUQ9Ikiogxz5JqwjMz8/OZubN4XAE8s+xQ0lSyZMkSHnroIfr6+njooYdYsmRJ2ZGkmjPRXT3/vdpBJO3b0UcfzQMPPDDm+P777y8xkaTCryPifwO9xfFiKpu9SJokq1ev5vTTT2d4eJi+vj5Wr3ZimrSnfRZ+ETGQma0RsZHKDdt3vwVkZh5W1XSSdrv//vt59rOfzcMPP8yznvUsiz6pdpwJfAb4VHH8H0WbpEkwY8YMNm7cyJlnnsm9997Lc5/7XDZu3MiMGRMa35CmjH1O9czM1uL50Mw8bNTjUIs+afI98sgj7Nq1i0ceeaTsKJIKmXlvZr4xM59ZPE7LzHvKziVNFcuWLWPLli3cd9997Nq1i/vuu48tW7Z4A3dpDxNd4wdARDwrIp478qhWKEnj2/MG7pLKFxHPi4jrIuKRYmfPb0TE88rOJU0VJ510Ek1NTWP6yKamJk466aSSk0m1ZaI3cH9jRNwO3AX8O3A38J0q5pIkqV58GbgaOIrKzp7X8Ph6P0lVdu655zJt2jRmzpwJwMyZM5k2bRrnnntuycmk2jLREb+PAa8AbsvM44FFVNYwSJpEhx9++JhnSTUhMvOLo3b1/H8Yuy5eUhUNDQ2xZcsWMit/dpnJli1bGBoaKjmZVFsmWvjtyMzfANMiYlpm9gEvqWIuSeN49NFHxzxLqgl9EXFeRMyPiOMi4lzgWxFxREQcUXY4aSoY7z5+ksaa6HZHGyLiEOAHwJci4mFgZ/ViSRrPtGnT2LVr1+5nSTXh7cXzX+7RfiaVkT/X+0mT4Nxzz2XBggWsW7eOD37wg2XHkWrORAu/U4HHgL8B3gE8Hfj7aoWSJKleFEsgJJVo2rRpnHfeeezYsWP3Gj83QpPGmugN3DcXL3cBV0bEdOB04EvVCibpiUZG+Rztk2pLRLQAC4DZI22Z+YXyEklTy/DwMIcddhiPPvoohxxyiEsipHHsc41fRBwWEedHxGci4k+j4mzgTuBtkxNRkqTaFREfBlYUj3bgH4E3lhpKmkKmTav85+ye6+BH2iVVPNlfxBeBFwA/B/4CuB54K3BqZp5a5WyS9jCyWN1F61JNeQuV3a4fysz3Ai8Gmvb3YhExNyK+GhG/jIjBiPgfxUYxN0TE7cWzW/tKhZHdPCfaLk1VT1b4PS8z35OZnwMWAwuB12fmT6ofTdKepk+fPuZZUk14LDN3ATsj4jDgYQ5sQ5dLgO9m5h9RKSIHgfOANZl5ArCmOJbE3kf2HPGTxnqyv4gdIy8ycxi4KzM3TuTCEXF5RDwcEbeOavtIRNwfET8pHq8b9d75EXFHRPwqIl7zVP+HSI1u+vTpY0b8LP6kmrE2IuYCq4CbgVuAH+3PhYrC8X8BPQCZuT0zN1DZZO3K4rQrgdMONLTUKEY2cdlzVoybu0hjPdnmLi+OiN8XrwN4WnEcQGbmYfv47BXAZ4A9F7d/KjP/aXRDRCygslnMC4HnAP8WEX9YFJuSqHRgI9NWhoeH3eBFqhGZeVbx8rKI+C5wWGb+bD8v9zzgEeDzEfFiKoXkXwHzMvPB4vsejIhnjffhiFgKLAWYN28e/f39+xlDqj8RQWbufgb8G5BG2Wfhl5n7PaSQmT+IiPkTPP1U4KrM3AbcFRF3AC8D/nN/v19qJCOd2J67errWTypfRHwD+Arwjcy8+wAvNwN4KdCZmT+MiEt4CtM6M3MlsBJg4cKF2dbWdoBxpPox3jp4/wakx030Pn4H09kR8S5gLfC3mfkocDRw06hzhoq2J/DXTE1F+1q47t+AVLqLqdzE/eMR8SMqReA3M3PrflxrCBjKzB8Wx1+lUvitj4ijitG+o6isI5Q0ykhf6aYu0vgmu/C7FPgYkMXzJ4EzqUwd3dO4f7X+mimN5d+AVK7M/Hfg34t73L4SWAJcDuxrOcTervVQRNwXES/IzF9R2S10XfF4N/CJ4vkbByu/1Ci81620b5Na+GXm+pHXEbEK+GZxOAQcO+rUY4AHJjGaJEn7LSKeBryBysjfS3l8I5b90Ql8KSJmUblv7nupbMZ2dUR0APdSubWSJEkTNqmF38g0leLwTcDIjp+rgS9HxMVUNnc5gf3cEU2SpMkUEV8BXg58F/hnoL+4vcN+KW6ZtHCctxbt7zUlSapa4RcRvUAbcGREDAEfBtoi4iVUpnHeDfwlQGb+IiKupjKVZSfwfnf0lJ5ovB3LJJXu88AZ9luSpFpWtcIvMxeP09yzj/O7ge5q5ZEagQvXpdoREedm5j9m5ncj4q3ANaOfd1DBAAAgAElEQVTe+4fMvKDEeJIkjfFkN3CXJEnjO33U6/P3eO+1kxlEEhx66KFMmzaNQw89tOwoUk2y8JPqyHj3KJJUmtjL6/GOJVXZxo0b2bVrFxs3biw7ilSTLPykOuJUT6mm5F5ej3csSVKpyriBuyRJjeDFEfF7KqN7TyteUxzPLi+WNLXsbcMzZ8dIY1n4SZK0HzJzetkZJO19FoyzY6SxnOopSZKkujdv3jwignnz5pUdRapJjvhJkiSp7q1fv37Ms6SxHPGT6shJJ53ENddcw0knnVR2FEmSJNURR/ykOnLjjTdy4403lh1DkqSaM7LJy942e5GmOkf8JEmSVPe85ZG0bxZ+kiRJktTgLPwkSZJU9w4//HBWrVrF4YcfXnYUqSa5xk+SJEl179FHH2XJkiVlx5BqliN+kiRJqmsRsc9jSRZ+kiRJqnN7bujiBi/SE1n4SZIkSVKDs/CTJEmSpAZn4SdJkiRJDc7CT5IkSZIanIWfJEmSJDU4Cz9JkmpMREyPiB9HxDeL4+Mj4ocRcXtEfCUiZpWdUZJUX6pW+EXE5RHxcETcOqrtiIi4oei4boiIw4v2iIhPR8QdEfGziHhptXJJklQH/goYHHV8IfCpzDwBeBToKCWVJKluVXPE7wrgtXu0nQesKTquNcUxwCnACcVjKXBpFXNJklSzIuIY4M+AfymOA3gl8NXilCuB08pJJ0mqVzOqdeHM/EFEzN+j+VSgrXh9JdAPLC/av5CVu23eFBFzI+KozHywWvkkSapR/xc4Fzi0OH4GsCEzdxbHQ8DR430wIpZS+QGVefPm0d/fX92kUo3zb0B6XNUKv72YN1LMZeaDEfGsov1o4L5R5410ahZ+kqQpIyJeDzycmTdHRNtI8zin5nifz8yVwEqAhQsXZltb23inSVOGfwPS4ya78NubCXdq/popjeXfgNRQTgbeGBGvA2YDh1EZAZwbETOKUb9jgAdKzChJqkOTXfitH5nCGRFHAQ8X7UPAsaPO22un5q+Z0lj+DUiNIzPPB84HKEb8PpiZ74iIa4C3AFcB7wa+UVpISVJdmuzbOaym0mHB2I5rNfCuYnfPVwC/c32fJEm7LQfOiYg7qKz56yk5jySpzlRtxC8ieqls5HJkRAwBHwY+AVwdER3AvcBbi9O/DbwOuAPYAry3WrkkSaoHmdlPZRM0MvNO4GVl5pEk1bdq7uq5eC9vLRrn3ATeX60skiRJkjSVTfZUT0mSJEnSJLPwkyRJkqQGZ+EnSZIkSQ3Owk+SJEmSGpyFnyRJkiQ1OAs/SZIkSWpwFn6SJEmS1OAs/CRJkiSpwVn4SZIkSVKDs/CTJEmSpAZn4SdJkiRJDc7CT5IkSZIanIWfJEmSJDU4Cz9JkiRJanAWfpIkSZLU4Cz8JEmSJKnBWfhJkiRJUoOz8JMkSZKkBmfhJ0mSJEkNzsJPkiRJkhqchZ8kSZIkNbhSCr+IuDsifh4RP4mItUXbERFxQ0TcXjwfXkY2SZLKEhHHRkRfRAxGxC8i4q+KdvtISdIBKXPErz0zX5KZC4vj84A1mXkCsKY4liRpKtkJ/G1mNgOvAN4fEQuwj5QkHaBamup5KnBl8fpK4LQSs0iSNOky88HMvKV4vREYBI7GPlKSdIBmlPS9CVwfEQl8LjNXAvMy80GodHwR8aySskmSVLqImA+cCPyQCfaREbEUWAowb948+vv7JyWrVKv8G5AeV1bhd3JmPlB0XDdExC8n+kE7NWks/wakxhMRhwBfA/46M38fERP6XPFD6kqAhQsXZltbW9UySvXAvwHpcaUUfpn5QPH8cER8HXgZsD4ijip+yTwKeHgvn7VTk0bxb0BqLBExk0rR96XM/NeieUJ9pCRJezPpa/wiYk5EHDryGvhT4FZgNfDu4rR3A9+Y7GySJJUpKkN7PcBgZl486i37SEnSASljxG8e8PVi2soM4MuZ+d2I+C/g6ojoAO4F3lpCNkmSynQy8E7g5xHxk6LtAuAT2EdKkg7ApBd+mXkn8OJx2n8DLJrsPJIk1YrMHAD2tqDPPlKStN9q6XYOkiRJkqQqsPCTJEmSpAZX1u0cJEmSpHFN9BYmB/M6mXlQvlOqVRZ+kiRJqilPpQjbV3FnMSc9zqmekiRJktTgLPwkSZJUt/Y2qudonzSWhZ8kSZLqWmaSmRy3/Ju7X0say8JPkiRJkhqchZ8kSZIkNTgLP0mSJElqcBZ+kiRJktTgLPwkSZIkqcFZ+EmSJElSg7PwkyRJkqQGN6PsAJIkSWo8L/7o9fzusR2T/r3zz/vWpH7f0582k59++E8n9Tul/WHhJ0mSpIPud4/t4O5P/Nmkfmd/fz9tbW2T+p2TXWhK+8upnpIkSZLU4Cz8JEmSJKnBOdVTkiRJB92hzefxoivPm/wvvnJyv+7QZoDJndIq7Q8LP0mSJB10Gwc/4Ro/qYY41VOSJEmSGpwjfpIkSaqKUkbDvjv5t3OQ6kHNFX4R8VrgEmA68C+Z+YmSI0mSVDr7R9WbyZ7mCZVCs4zvlepBTU31jIjpwD8DpwALgMURsaDcVJIklcv+UZJ0oGqq8ANeBtyRmXdm5nbgKuDUkjNJVRERE34crOs82bUk1Sz7R0nSAam1qZ5HA/eNOh4CXj76hIhYCiwFmDdvHv39/ZMWThpP5z2d+/W5litaDnKSiXnRlS/ar8+tOG7FQU4i6Sl40v4R7CPVONrb2/f7s3Hh/n2ur69vv79Tqge1VviNNxyRYw4yVwIrARYuXJiTvWWvtKef8/Oqf8e+Ruoyc6/vSWoYT9o/gn2kGsf+9m1l3M5Bqhe1NtVzCDh21PExwAMlZZFqxt46QIs+acqwf5QkHZBaK/z+CzghIo6PiFnA6cDqkjNJNSEzyUz6+vp2v5Y0Zdg/SpIOSE1N9czMnRFxNvA9KttVX56Zvyg5liRJpbJ/lCQdqJoq/AAy89vAt8vOIUlSLbF/lCQdiFqb6ilJkiRJOsgs/CRJkiSpwVn4SZIkSVKDs/CTJEmSpAZn4SdJkiRJDS7q+V5gEfEIcE/ZOaRJdiTw67JDSCU4LjOfWXaIemEfqSnKPlJT0YT6x7ou/KSpKCLWZubCsnNIklRr7COlvXOqpyRJkiQ1OAs/SZIkSWpwFn5S/VlZdgBJkmqUfaS0F67xkyRJkqQG54ifJEmSJDU4Cz9JkiRJanAWftIkiojhiPhJRNwaEddFxNwJfObGCZzzPyPiF8W1n7aP8zYVz/Mj4tanll6SpIkb1eeNPM4rO9NoEfGSiHjdqOM31lpG6WByjZ80iSJiU2YeUry+ErgtM7sPwnUvA36YmZ+fyPdHxHzgm5nZcqDfLUnSeEb3eSVmmJGZO/fy3nuAhZl59uSmksrhiJ9Unv8EjgaIiEMiYk1E3BIRP4+IU0dOGjVK1xYR/RHx1Yj4ZUR8KSr+Angb8HdF216vJUlSmSLilIi4etRxW0RcV7y+NCLWFjNYPjrqnLsj4sKI+FHxeH7RflzR3/2seH5u0X5FRFwcEX3AhRHxsoi4MSJ+XDy/ICJmAX8PvL0YjXx7RLwnIj4zgWt/urjOnRHxlkn7P086QDPKDiBNRRExHVgE9BRNW4E3ZebvI+JI4KaIWJ1PHJI/EXgh8ADwH8DJmfkvEdFKZQTvqxExY4LXkiSpmp4WET8Zdfxx4GvA5yJiTmZuBt4OfKV4vyszf1v0kWsi4o8z82fFe7/PzJdFxLuA/wu8HvgM8IXMvDIizgQ+DZxWnP+HwKsyczgiDgP+V2bujIhXAf+QmW+OiL9j1IhfMQI4Yl/XPgpoBf4IWA189SD8fyVVnSN+0uQa6QR/AxwB3FC0B/APEfEz4N+ojATOG+fzP8rMoczcBfwEmD/OORO9liRJ1fRYZr5k1OMrxbTL7wJvKH6o/DPgG8X5b4uIW4AfU/mRc8Goa/WOev4fxev/AXy5eP1FKsXYiGsyc7h4/XTgmmJt+6eKaz+ZfV372szclZnrsH9VHbHwkybXY5n5EuA4YBbw/qL9HcAzgT8p3l8PzB7n89tGvR5m/FH7iV5LkqQyfIXKEoVXAv+VmRsj4njgg8CizPxj4FuM7btyL6/ZS/vmUa8/BvQV69rfwP71iaOvPbovjv24llQKCz+pBJn5O+ADwAcjYiaVXyMfzswdEdFOpTDcXwfzWpIkHWz9wEuBJTw+zfMwKsXa7yJiHnDKHp95+6jn/yxe3wicXrx+BzCwl+97OnB/8fo9o9o3Aofu5TMTvbZUN1zjJ5UkM38cET+l0rF8CbguItZSmcL5ywO49MG8liRJ+2vPNX7fzczzinV336RShL0bIDN/GhE/Bn4B3EllHftoTRHxQyqDFouLtg8Al0fEh4BHgPfuJcc/AldGxDnA90e19wHnFRk/vsdnJnptqW54OwdJkiTVrIi4m8omLL8uO4tUz5zqKUmSJEkNzhE/SZIkSWpwjvhJkiRJUoOz8JMkSZKkBmfhJ0mSJEkNzsJPkiRJkhqchZ8kSZIkNTgLP0mSJElqcBZ+kiRJktTgLPwkSZIkqcFZ+EmSJElSg7PwkyRJkqQGZ+EnSZIkSQ3Owk+SJEmSGpyFnyRJkiQ1OAs/SZIkSWpwFn6SJEmS1OAs/CRJkiSpwVn4SZIkSVKDs/CTJEmSpAZn4SdJkiRJDc7CT5IkSZIanIWfJEmSJDU4Cz9JkiRJanAWfpIkSZLU4Cz8JEmSJKnBWfhJkiRJUoOz8JMkSZKkBmfhJ0mSJEkNzsJPkiRJkhqchZ8kSZIkNTgLP0mSJElqcBZ+kiRJktTgZpQd4EAceeSROX/+/LJjSJNq8+bNzJkzp+wY0qS7+eabf52Zzyw7R72wj9RUZB+pqWii/WNdF37z589n7dq1ZceQJlV/fz9tbW1lx5AmXUTcU3aGemIfqanIPlJT0UT7R6d6SpIkSVKDs/CTJEmSpAZn4SdJkiRJDc7CT5IkSZIanIWfJEmSJDU4Cz9JkiRJanAWflKd6O3tpaWlhUWLFtHS0kJvb2/ZkSRJqgn2kdKTq+v7+ElTRW9vL11dXfT09DA8PMz06dPp6OgAYPHixSWnkySpPPaR0sQ44ifVge7ubnp6emhvb2fGjBm0t7fT09NDd3d32dEkSSqVfaQ0MRZ+Uh0YHByktbV1TFtrayuDg4MlJZIkqTbYR0oTY+En1YHm5mYGBgbGtA0MDNDc3FxSIkmSaoN9pDQxFn5SHejq6qKjo4O+vj527txJX18fHR0ddHV1lR1NkqRS2UdKE+PmLlIdGFmc3tnZyeDgIM3NzXR3d7toXZI05dlHShMTmVl2hv22cOHCXLt2bdkxpEnV399PW1tb2TGkSRcRN2fmwrJz1Av7SE1F9pGaiibaPzrVU5IkSZIanIWfJEmSJDU4Cz9JkiTVtc7OTmbPnk17ezuzZ8+ms7Oz7EhSzXFzF0mSJNWtzs5OLrvsMi688EIWLFjAunXrWL58OQArVqwoOZ1UOxzxkyRJUt1atWoVF154Ieeccw6zZ8/mnHPO4cILL2TVqlVlR5NqioWfJEmS6ta2bdtYtmzZmLZly5axbdu2khJJtcnCT5IkSXWrqamJyy67bEzbZZddRlNTU0mJpNrkGj9JkiTVrSVLluxe07dgwQIuvvhili9f/oRRQGmqs/CTJElS3RrZwOWCCy5g27ZtNDU1sWzZMjd2kfZQtameEXF5RDwcEbeOarsoIn4ZET+LiK9HxNxR750fEXdExK8i4jXVyiVJkqTGsmLFCrZu3UpfXx9bt2616JPGUc01flcAr92j7QagJTP/GLgNOB8gIhYApwMvLD7z2YiYXsVskiTVpIj4m4j4RUTcGhG9ETE7Io6PiB9GxO0R8ZWImFV2TklSfala4ZeZPwB+u0fb9Zm5szi8CTimeH0qcFVmbsvMu4A7gJdVK5skSbUoIo4GPgAszMwWYDqVH0YvBD6VmScAjwId5aWUJNWjMtf4nQl8pXh9NJVCcMRQ0fYEEbEUWAowb948+vv7qxhRqj2bNm3y373U2GYAT4uIHcAfAA8CrwTOKN6/EvgIcGkp6SRJdamUwi8iuoCdwJdGmsY5Lcf7bGauBFYCLFy4MNva2qoRUapZ/f39+O9eakyZeX9E/BNwL/AYcD1wM7Bh1IwZfxyV9sIfR6W9m/TCLyLeDbweWJSZI8XdEHDsqNOOAR6Y7GySJJUpIg6nsvzheGADcA1wyjin+uOoNA5/HJX2blJv4B4RrwWWA2/MzC2j3loNnB4RTRFxPHAC8KPJzCZJUg14FXBXZj6SmTuAfwVOAuZGxMiPtf44Ku2ht7eXlpYWFi1aREtLC729vWVHkmpO1Ub8IqIXaAOOjIgh4MNUdvFsAm6ICICbMnNZZv4iIq4G1lGZAvr+zByuVjZJkmrUvcArIuIPqEz1XASsBfqAtwBXAe8GvlFaQqnG9Pb20tXVRU9PD8PDw0yfPp2Ojsr+R4sXLy45nVQ7qrmr5+LMPCozZ2bmMZnZk5nPz8xjM/MlxWPZqPO7M/O/ZeYLMvM71colSVKtyswfAl8FbgF+TqWfXklltsw5EXEH8Aygp7SQUo3p7u6mp6eH9vZ2ZsyYQXt7Oz09PXR3d5cdTaopZe7qKUmS9pCZH6YyS2a0O/E2R9K4BgcHaW1tHdPW2trK4OBgSYmk2jSpa/wkSZKkg6m5uZmBgYExbQMDAzQ3N5eUSKpNjvhJkiSpbnV1dXHqqaeydetWduzYwcyZM5k9ezaf+9znyo4m1RRH/CRJklS3brzxRjZv3swRRxwBwBFHHMHmzZu58cYbS04m1RYLP0mSJNWtVatWcdFFF/HQQw/R19fHQw89xEUXXcSqVavKjibVFAs/SZIk1a1t27axbNmyMW3Lli1j27ZtJSWSapOFnyRJkupWU1MTl1122Zi2yy67jKamppISSbXJzV0kSZJUt5YsWcLy5csBWLBgARdffDHLly9/wiigNNVZ+EmSJKlurVixgttuu40PfvCDZCYRwatf/WpWrFhRdjSppjjVU5IkSXWrt7eX22+/nTVr1nDDDTewZs0abr/9dnp7e8uOJtUUCz9JkiTVre7ubnp6emhvb2fGjBm0t7fT09NDd3d32dGkmmLhJ0mSpLo1ODhIa2vrmLbW1lYGBwdLSiTVJgs/SZIk1a3m5mYGBgbGtA0MDNDc3FxSIqk2ubmLJEmS6lZXVxennnoqW7duZceOHcycOZPZs2fzuc99ruxoUk1xxE+SJEl168Ybb2Tz5s0cccQRABxxxBFs3ryZG2+8seRkUm2x8JMkSVLdWrVqFRdddBEPPfQQfX19PPTQQ1x00UWsWrWq7GhSTbHwkyRJUt3atm3bE27WvmzZMrZt21ZSIqk2WfhJkiSpbjU1NXHZZZeNabvssstoamoqKZFUm9zcRZIkSXVryZIlLF++HIAFCxZw8cUXs3z58ieMAkpTnYWfJEmS6taKFSsAuOCCC9i2bRtNTU0sW7Zsd7ukCqd6SpIkqa7ddtttbN++HYDt27dz2223lZxIqj0WfpIkSapbr3nNa7j++utZtmwZ1113HcuWLeP666/nNa95TdnRpJriVE9JkiTVrRtuuIH3ve99fPazn6W/v5/PfvazAE/Y8EWa6hzxkyRJUt3KTD7+8Y+Pafv4xz9OZpaUSKpNFn6SJEmqWxHB+eefP6bt/PPPJyJKSiTVpqoVfhFxeUQ8HBG3jmo7IiJuiIjbi+fDi/aIiE9HxB0R8bOIeGm1ckmSJKlxvPrVr+bSSy/lrLPOYtOmTZx11llceumlvPrVry47mlRTqjnidwXw2j3azgPWZOYJwJriGOAU4ITisRS4tIq5JEmS1CC+973v8aIXvYhLL72UN7zhDVx66aW86EUv4nvf+17Z0aSaUrXCLzN/APx2j+ZTgSuL11cCp41q/0JW3ATMjYijqpVNkiRJjaG3t5dNmzbx/e9/nxtuuIHvf//7bNq0id7e3rKjSTVlstf4zcvMBwGK52cV7UcD9406b6hokyRJkvaqu7ubnp4e2tvbmTFjBu3t7fT09NDd3V12NKmm1MrtHMZbfTvuVkwRsZTKdFDmzZtHf39/FWNJtWfTpk3+u5ckqTA4OEhra+uYttbWVgYHB0tKJNWmyS781kfEUZn5YDGV8+GifYj/v737j7KrKg8+/n0cYhIVwy8z5QU0CFQGEFFHNEDtjBSWLRQiL1RTbRFT0qxqQLGVSGjRFlihWn5pa0yMvOmrQpWKoaCITefqiiAtPyIBBoEir0YjaAUkCwwhPu8f9wxMkvlxMzP3njM3389ad83d+5579jNZc3l47t5nH9hv0HH7Aj8d6gSZuQxYBtDd3Z09PT1NDFeqnlqthn/3kiTVdXV1sWbNGnp7e5/vW7NmDV1dXSVGJVVPqwu/64HTgSXFz1WD+j8QEdcAbwaeHFgSKkmSJA1n8eLFvO1tb9uu/0tf+lIJ0UjV1czbOVwN3Aq8JiLWR8Q86gXfcRHxIHBc0Qb4OvAw8BCwHPiLZsUlSZKk9vHHf/zHO9Qv7ayaNuOXmXOHeenYIY5N4P3NikWSJEntLTOfvxzCm7dL22v1rp6SJEnShPra1742YluShZ8kSZImuTlz5ozYllSd2zlIkiRJY+byTmlkzvhJkiRJUpuz8JMkSdKk1tHRQWbS19dHZtLR0VF2SFLlWPhJkiRpUlu9evWIbUkWfpIkSZrkjj322BHbktzcRZIkSZPcli1b3NxFGoUzfpIkSZLU5iz8JEmqkIjYLSKujYj7I6I/ImZHxB4R8a2IeLD4uXvZcUpVM3hzF0nbs/CTJKlargBuysyDgdcB/cAiYHVmHgSsLtqSCtdee+2IbUkWfpIkVUZEvBx4K7ACIDOfzcwngJOBlcVhK4E55UQoVdOpp546YluSm7tIklQlrwZ+DlwVEa8D7gDOBjozcwNAZm6IiJklxihVkpu7SCOz8JMkqTp2Ad4ALMzM2yLiCnZgWWdEzAfmA3R2dlKr1ZoSpDRZ+BmQXmDhJ00SCxcuZPny5WzatImpU6dy5pln8qlPfarssCRNrPXA+sy8rWhfS73wezQi9i5m+/YGHhvqzZm5DFgG0N3dnT09PS0IWaqGzKRWq9HT0/P87J+fAekFXuMnTQILFy5k6dKlXHzxxXzjG9/g4osvZunSpSxcuLDs0CRNoMz8GfDjiHhN0XUscB9wPXB60Xc6sKqE8KTK+sIXvjBiW5KFnzQpLF++nEsuuYRzzjmHadOmcc4553DJJZewfPnyskOTNPEWAl+MiLuBI4CLgSXAcRHxIHBc0ZZUeM973jNiW5JLPaVJYdOmTSxYsGCrvgULFvDhD3+4pIgkNUtmrgW6h3jp2FbHIk0mbu4ijcwZP2kSmDp1KkuXLt2qb+nSpUydOrWkiCRJkjSZNFT4RcSJEXFXRPwyIn4VEU9FxK+aHZykujPPPJNzzz2XSy+9lF//+tdceumlnHvuuZx55pllhyZpCOZNqfUyk76+PjKz7FCkSmp0qeflwCnAuvTTJLXcwO6d55133vO7ei5YsMBdPaXqMm9KLeZST2lkjS71/DFwj8lLKs9RRx3FgQceyIte9CIOPPBAjjrqqLJDkjQ886ZUgiOOOKLsEKTKanTG7yPA1yPi28Cmgc7MvLQpUUnaytVXX83ixYtZsWIFW7ZsoaOjg3nz5gEwd+7ckqOTNATzplSCN7/5zaxdu7bsMKRKanTG7yLgaWAasOugh6QWuOiii1ixYgW9vb3ssssu9Pb2smLFCi666KKyQ5M0NPOmVILPfvazZYcgVVajM357ZObxTY1E0rD6+/s55phjtuo75phj6O/vLykiSaMwb0otlpnUajV6enq83k8aQqMzfv8eESYwqSRdXV2sWbNmq741a9bQ1dVVUkSSRmHelFosIujt7bXok4bRaOH3fuCmiHhmIraljogPRcS9EXFPRFwdEdMiYv+IuC0iHoyIf4mIF4/1/FK7Wbx4MfPmzaOvr4/nnnuOvr4+5s2bx+LFi8sOTdLQBvLmr4uc6e0cJEmlamipZ2ZO2HUJEbEPcBZwSGY+ExFfBt4F/AFwWWZeExFLgXnAZyZqXGkyG9jAZeHChfT399PV1cVFF13kxi5SRU1k3pTUGJd6SiNrdMaPiNg9Io6MiLcOPMYx7i7A9IjYBXgJsAF4G3Bt8fpKYM44zi+1nblz53LPPfewevVq7rnnHos+qeIi4pSIuDQi/iEizGlSE/X29o7YltTgjF9E/BlwNrAvsBZ4C3Ar9WJth2TmTyLik8CPgGeAm4E7gCcy87nisPXAPsPEMh+YD9DZ2UmtVtvREKRJbePGjf7dSxUXEf8EHAhcXXQtiIjjMvP9JYYlta2+vr4R25Ia39XzbOBNwPcyszciDgY+PpYBI2J34GRgf+AJ4CvA7w9x6JA3vc3MZcAygO7u7uzp6RlLGNKkNbCMRVKl/S5w2MAN3CNiJbCu3JCk9ubyTmlkjS71/HVm/hogIqZm5v3Aa8Y45u8BP8zMn2fmZuCrwFHAbsXST6jPLP50jOeXJKlsPwBeOai9H3B3SbFIktRw4bc+InYDvgZ8KyJWMfbC7EfAWyLiJVH/auZY4D6gDzi1OOZ0YNUYzy9JUtn2BPojohYRNep57hURcX1EXF9uaFJ7ykz6+vooJtolbaPRXT3fUTz9WET0ATOAm8YyYGbeFhHXAncCzwF3UV+6eSNwTURcWPStGMv5JUmqgL8pOwBpZzJr1qzt2o888kgpsUhV1ejmLq+gvvzyOeCOzNw4nkEz8wLggm26HwaOHM95JUmqgsz89sDziNgjM39ZZjxSu9u2yLPok7Y3YuEXEYcAVwKzqF+rcBcwMyK+DZydmU82PUJJkiaJiDga+BzwG+B9wIXAARExBfijzLy1zPikdubmLtLIRrvG7/PA+zPzQOAY4P7M3B/4Li7FlCRpW69g4jgAABg6SURBVJcBfwT8GfVLGD6ema+mvpv1J8sMTJK0cxut8JuemT8AyMz/BF5bPF8OHNLk2CQNcvjhhxMR9Pb2EhEcfvjhZYckaXtTMnNdMbP388xcA5CZdwLTyw1Nam9u7iKNbLTC778j4q8j4qjiputrAYolK43eA1DSOB1++OGsW7eOk046ieuuu46TTjqJdevWWfxJ1TM4r350m9de3MpAJEkabLTC733ArsB5wCbqN3IHeAnwp02MS9IgA0XfqlWr2G233Vi1atXzxZ+kSvnriHgJQGZ+baAzIg4A/rm0qCRJO70RZ+0y8wngI0P0Pwl8r1lBSdreCSecwGGHHUZ/fz9dXV2cddZZXH+9twOTqiQzh/xQZuZ/A3/f4nCknYqbu0gjG21Xz38Dhl0onZknTXhEkob0oQ99iBtuuIEtW7bQ0dHBiSeeWHZIkrZh3pRaLzOHLPq81k/a2mhLPT8J/APwQ+AZYHnx2Ajc09zQJA2YOnUqTz/9NJdffjkbN27k8ssv5+mnn2bq1KllhyZpa+ZNqcWGm+lzBlDa2mhLPb8NEBF/l5lvHfTSv0XEd5oamaTnbd68mcMOO4zrr7/++eWdhx12GPfdd1/JkUkazLwplSczqdVq9PT0WPRJQxhtxm/AKyLi1QONiNgfeEVzQpK0ra6uLq688sqttqq+8sor6erqKjs0SUMzb0qSKqXRWzJ8CKhFxMNFexbw502JSNJ2Fi9ezJw5c3jmmWfYvHkzU6ZMYfr06SxdurTs0CQNzbwpSaqUhgq/zLwpIg4CDi667s/MTc0LS9Jgt9xyCxs3bmTmzJk89thj7Lnnnjz22GPccsstzJ07t+zwJG3DvCm1nss7pZE1tNSzuCfRXwEfyMzvA6+MCLcUlFpk+fLlfOITn2DDhg2sXr2aDRs28IlPfILly5eXHZqkIZg3JUlV0+g1flcBzwKzi/Z64MKmRCRpO5s2bWLBggVb9S1YsIBNm5xAkCrKvCm12ODr4CVtr9Fr/A7IzHdGxFyAzHwmnE+XWmbq1Km8/vWv58EHH3z+fkUHHXSQt3OQqsu8KUmqlEZn/J6NiOkUN6WNiAMApxqkFpk5cyYPPPAAs2fP5itf+QqzZ8/mgQceYObMmWWHJmlo5k1JUqU0OuN3AXATsF9EfBE4Gnhvs4KStLX169dz6KGHcscdd3DaaacxdepUDj30UO/jJ1WXeVNqMSfVpZE1uqvntyLiTuAtQABnZ+YvmhqZpOdlJg899NDz1/Rt2rSJhx56yOsYpIoyb0qSqqbRXT0D+H3gjZl5A/CSiDiyqZFJ2sqmTZvo7OzkqquuorOz041dpAozb0qt5+Yu0sgavcbvn6jvTDZww7CngH9sSkSShnXKKaew1157ccopp5QdiqSRmTelFosIent7XfIpDaPRa/zenJlviIi7ADLz8Yh4cRPjkrSN2bNns3TpUj7zmc8QEcyePZtbb7217LAkDc28KUmqlEYLv80R0cELu5O9AvhN06KStJ3BRV5mWvRJ1WbelCRVSqNLPa8ErgM6I+IiYA1wcdOikjSsiy66qOwQJI3OvClJqpRGd/X8YkTcARxbdM3JzP7mhSVpOIsXLy47BEmjMG9KrZeZ1Go1enp6vM5PGkKjSz0BXgIMLFuZ3pxwJElqG+ZNqYUs9qSRNXo7h78BVgJ7AHsBV0XE+WMdNCJ2i4hrI+L+iOiPiNkRsUdEfCsiHix+7j7W80vtatq0aXz6059m2rRpZYciaQQTnTclSRqvRq/xmwu8KTM/lpkXUL8h7bvHMe4VwE2ZeTDwOqAfWASszsyDgNVFW9IgM2bM4KUvfSkzZswoOxRJIxtX3oyIjoi4KyJuKNr7R8RtxZej/+IOodL2vI+fNLJGC79HgMFTDFOB/x7LgBHxcuCtwAqAzHw2M58ATqb+7SjFzzljOb/Urjo6Onj00Uc544wzePTRR+no6Cg7JEnDe4Tx5c2zqX8pOuAS4LLiy9HHgXnjDVCStHNptPDbBNwbEf8nIq4C7gE2RsSVEXHlDo75auDn1Je93BURn4uIlwKdmbkBoPg5cwfPK7W1LVu2jNiWVCljzpsRsS9wAvC5oh3A24Bri0P8clSStMMa3dzluuIxoDbOMd8ALMzM2yLiCnZgWWdEzAfmA3R2dlKrjScUafI5//zzufDCC59v+xmQKmk8efNy4CPArkV7T+CJzHyuaK8H9hnqjeZI7cyG2tzFz4D0gtiRddARMQU4DPhJZj42pgEjfgv4XmbOKtq/Q73wOxDoycwNEbE3UMvM14x0ru7u7rz99tvHEoY0qYy0U5nXMmhnERF3ZGZ32XHsiB3NmxFxIvAHmfkXEdED/CVwBnBrZh5YHLMf8PXMfO1I5zJHamcyVJ40P2pn0Wh+HHGpZ0QsjYhDi+czgO8D/wzcFRFzxxJYZv4M+HFEDBR1xwL3AdcDpxd9pwOrxnJ+SZLKMgF582jgpIh4BLiG+hLPy4HdImJglc6+wE8nOnZpMsvMrTZ3seiTtjfaNX6/k5n3Fs/PAB4ovmF8I/VlKGO1EPhiRNwNHAFcDCwBjouIB4HjirakbZx/vjvCSxU2rryZmR/NzH2LVTHvAv4jM98N9AGnFof55ajaXkSM6dHb2zvm90rtbrRr/J4d9Pw44CtQn7UbzwckM9cCQ01HHjvmk0o7icHX90mqnKbkTeBc4JqIuBC4i2JnbKldjXXGbtaiG3lkyQkTHI3UHkYr/J4orjf4CfXlJ/MAiuUm05scmyRJk82E5c3MrFFsCpOZDwNHTmSgkqSdy2hLPf8c+ABwFfDB4vo8qM/M3djMwCRtz5vTSpVn3pQkVdKIM36Z+QDw9iH6vwl8s1lBSRqa1yBI1WbelCRV1YiFX0R8Chh2aiEzz5rwiCRJmqTMm5KkqhptqeftwB3ANOo3XX+weBwBbGluaJKGcvzxx5cdgqThmTclSZU02lLPlQAR8V6gNzM3F+2lwM1Nj07Sdm6+2Y+eVFXmTUlSVY024zfgfwG7Dmq/rOiTJEnbM29KkipltNs5DFgC3BURfUX7d4GPNSUiSSPq6OhgyxZXjEkVZ96UJFVKQ4VfZl4VEd8A3lx0LRq0RbWkFrLok6rPvClJqppGl3oCdAA/Bx4Hfjsi3tqckCRJagvmTUlSZTQ04xcRlwDvBO4FflN0J/CdJsUlaRi77747jz/+eNlhSBqBeVOSVDWNXuM3B3hNZm5qZjCSRtfZ2WnhJ1WfeVOSVCmNLvV8GJjSzEAkNeb+++8vOwRJozNvSpIqpdEZv6eBtRGxGnj+28vMPKspUUmSNLmZNyVJldJo4Xd98ZAkSaMzb0qSKqXR2zmsbHYgkiS1C/OmJKlqRiz8IuLLmflHEbGO+m5kW8nMw5sWmSRJk4x5U5JUVaPN+N0VEW8C3gFsbkE8kiRNZuZNSVIljVb47QlcARwM3A3cAnwXuDUzf9nk2CQNYY899uCXv/TjJ1WUeVOSVEkjFn6Z+ZcAEfFioBs4CngfsDwinsjMQ5ofoqTBLPqk6jJvSpKqqtFdPacDLwdmFI+fAuuaFZQkSZOceVOSVCmjbe6yDDgUeAq4jfqSlUsz8/EWxCZJ0qRi3pQkVdWLRnn9lcBU4GfAT4D1wBPNDkqSpEnKvClJqqQRC7/MfDvwJuCTRdeHgf+KiJsj4uPNDk7S1jKTvr4+MrfbJV5SBZg3JUlVNeo1fln/P8x7IuIJ4MnicSJwJHBBc8OTJGlyMW9KkqpotGv8zqK+I9nR1O9H9F3gVuDzeJG61HIRUXYIkkZg3pQkVdVoM36zgGuBD2XmhokcOCI6gNuBn2TmiRGxP3ANsAdwJ/AnmfnsRI4pSVKTzaJJeVOSpPEY7Rq/czLz2iYlr7OB/kHtS4DLMvMg4HFgXhPGlCSpaZqcNyVJGrPRdvVsiojYFzgB+FzRDuBt1L8lBVgJzCkjNkmSJElqN43ewH2iXQ58BNi1aO8JPJGZzxXt9cA+Q70xIuYD8wE6Ozup1WrNjVSqOD8DkiRJGk3LC7+IOBF4LDPviIiege4hDh1yv/rMXAYsA+ju7s6enp6hDpN2Gn4GJEmSNJoyZvyOBk6KiD8ApgEvpz4DuFtE7FLM+u0L/LSE2CRJkiSp7bT8Gr/M/Ghm7puZs4B3Af+Rme8G+oBTi8NOB1a1Ojap6ryBuyRJksairGv8hnIucE1EXAjcBawoOR6pcryPnyRJksai1MIvM2tArXj+MHBkmfFIVZWZQxZ9zvxJkiSpEVWa8ZN2KhMxezeWc1gsSpIk7XxKuY+fpHoBNpbHq869YczvteiTJEnaOTnjJ0mSpAn3uo/fzJPPbG75uLMW3djS8WZMn8L3Lzi+pWNKY2HhJ0mSpAn35DObeWTJCS0ds1artfz+tq0uNKWxcqmnJEmSJLU5Cz9JkiRJanMWfpIkSZLU5iz8JEmSJKnNWfhJklQREbFfRPRFRH9E3BsRZxf9e0TEtyLiweLn7mXHKkmaXCz8JEmqjueAD2dmF/AW4P0RcQiwCFidmQcBq4u2JEkNs/CTJKkiMnNDZt5ZPH8K6Af2AU4GVhaHrQTmlBOhJGmysvCTJKmCImIW8HrgNqAzMzdAvTgEZpYXmSRpMvIG7pIkVUxEvAz4V+CDmfmriGj0ffOB+QCdnZ3UarWmxSg1otV/gxs3bizl797PmiYDCz9JkiokIqZQL/q+mJlfLbofjYi9M3NDROwNPDbUezNzGbAMoLu7O3t6eloRsjS0m26k1X+DtVqt5WOW8XtKY+FST0mSKiLqU3srgP7MvHTQS9cDpxfPTwdWtTo2SdLk5oyfJEnVcTTwJ8C6iFhb9J0HLAG+HBHzgB8Bp5UUn9SwXbsW8dqVJWxAu3L0QybSrl0AJ7R2UGkMLPwkSaqIzFwDDHdB37GtjEUar6f6l/DIktYWRGUs9Zy16MaWjieNlUs9JUmSJKnNWfhJkiRJUpuz8JMkSZKkNmfhJ0mSJEltzsJPkiRJktqchZ8kSZIktTkLP0mSJElqcxZ+kiRJktTmLPwkSZIkqc21vPCLiP0ioi8i+iPi3og4u+jfIyK+FREPFj93b3VskiRJktSOypjxew74cGZ2AW8B3h8RhwCLgNWZeRCwumhLkiRJksap5YVfZm7IzDuL508B/cA+wMnAyuKwlcCcVscmSZIkSe1olzIHj4hZwOuB24DOzNwA9eIwImYO8575wHyAzs5OarVaS2KVqsS/e0mSJO2I0gq/iHgZ8K/ABzPzVxHR0PsycxmwDKC7uzt7enqaFqNUSTfdiH/3kqTJYNaiG1s/6E2tHXPG9CktHU8aq1IKv4iYQr3o+2JmfrXofjQi9i5m+/YGHisjNmlHve7jN/PkM5tbOmarE+mM6VP4/gXHt3RMSdLk9siSE1o+5qxFN5YyrjQZtLzwi/rU3gqgPzMvHfTS9cDpwJLi56pWxyaNxZPPbG5pkqnVai2f8SvlG1tJkiRNmDJm/I4G/gRYFxFri77zqBd8X46IecCPgNNKiE2SJEmS2k7LC7/MXAMMd0Hfsa2MRZIkSZJ2BmXcx0+SJEmS1EIWfpIkSZLU5iz8JEmSJKnNWfhJkiRJUpuz8JMkSZKkNmfhJ0mSJEltroz7+EltZdeuRbx25aLWDrqytcPt2gXQupvUS5IkaWJZ+Enj9FT/Eh5Z0rqiqFar0dPT07LxAGYturGl40mSJGliudRTkiRJktqchZ8kSZIktTkLP0mSJElqcxZ+kiRJktTmLPwkSZIkqc1Z+EmSJElSm7PwkyRJkqQ2Z+EnSZIkSW3OG7hLE6DlNzi/qbXjzZg+paXjSZIkaWJZ+Enj9MiSE1o63qxFN7Z8TEmSJE1uLvWUJEmSpDZn4SdJkiRJbc7CT5IkSZLanIWfJEmSJLU5Cz9JkiRJanMWfpIkSZLU5iz8JEmSJKnNVa7wi4i3R8QPIuKhiFhUdjySJFWB+VGSNB6VKvwiogP4R+D3gUOAuRFxSLlRSZJULvOjJGm8KlX4AUcCD2Xmw5n5LHANcHLJMUmSVDbzoyRpXKpW+O0D/HhQe33RJ0nSzsz8KEkal13KDmAbMURfbnVAxHxgPkBnZye1Wq0FYUkTr7e3d8zvjUvGPm5fX9/Y3yypLKPmRzBHqn2UkSPNj2p3VSv81gP7DWrvC/x08AGZuQxYBtDd3Z09PT0tC06aSJnb/T9bQ2q1Gv7dSzudUfMjmCPVPsyR0sSr2lLP/wIOioj9I+LFwLuA60uOSZKkspkfJUnjUqkZv8x8LiI+AHwT6AA+n5n3lhyWJEmlMj9KksarUoUfQGZ+Hfh62XFIklQl5kdJ0nhUbamnJEmSJGmCWfhJkiRJUpuz8JMkSZKkNmfhJ0mSJEltzsJPkiRJktqchZ8kSZIktbnIzLJjGLOI+Dnw/8qOQ2qxvYBflB2EVIJXZeYryg5isjBHaidljtTOqKH8OKkLP2lnFBG3Z2Z32XFIklQ15khpeC71lCRJkqQ2Z+EnSZIkSW3Owk+afJaVHYAkSRVljpSG4TV+kiRJktTmnPGTJEmSpDZn4ScNISIui4gPDmp/MyI+N6j9DxFxXkRcu4PnfW9EfLp4/pqIqEXE2ojoj4imLk+JiJ6IuKF4vntEXBcRd0fEf0bEYc0cW5LUPnaCHHlykR/XRsTtEXFMM8eWWsXCTxraLcBRABHxIur3BTp00OtHAasz89RxjHElcFlmHpGZXcCnxnGuHXUesDYzDwf+FLiihWNLkia3ds+Rq4HXZeYRwPuAz41yvDQpWPhJQ/suRVKjnszuAZ4qZsqmAl3A4xFxDzz/LeVXI+KmiHgwIv5+4EQRcUZEPBAR3waOHjTG3sD6gUZmrht0rlXFuX4QERcMOtd7ihm6tRHx2YjoKPqPj4hbI+LOiPhKRLys6H97RNwfEWuAUwaNfQj1xEZm3g/MiojO4j1fi4g7IuLeiJg/aOyNEXFJ8dq/R8SRxbexD0fESeP615YkTSZtnSMzc2O+sAnGS4Esju+JiO8UK2bui4ilReFrjtSkYOEnDSEzfwo8FxGvpJ7cbgVuA2YD3cDdwLPbvO0I4J3Aa4F3RsR+EbE38HHqyew46gXXgMuA/4iIb0TEhyJit0GvHQm8uzjnaRHRHRFdxfmPLr6F3AK8OyL2As4Hfi8z3wDcDpwTEdOA5cAfAr8D/Nag83+fIslFxJHAq4B9i9fel5lvLH7PsyJiz6L/pUCteO0p4MLid3oH8LeN/LtKkia/nSBHEhHviIj7gRupz/oNHvvDxe9xAC8UjOZIVd4uZQcgVdjAN5pHAZcC+xTPn6S+zGVbqzPzSYCIuI96MbUX9UTw86L/X4DfBsjMqyLim8DbgZOBP4+I1xXn+lZm/k/xnq8CxwDPAW8E/isiAKYDjwFvoZ4sv1v0v5h6Ej4Y+GFmPlic5wvAwAzeEuCKiFgLrAPuKs4P9WLvHcXz/YCDgP+hnsRvKvrXAZsyc3NErANmNfQvKklqF+2cI8nM64DrIuKtwN8Bv1e89J+Z+XDxnquLsa/FHKlJwMJPGt7ANQyvpb6M5cfUv+X7FfD5IY7fNOj5Fl74fA17z5TiW9PPA58vlsQcNsx7EghgZWZ+dPALEfGH1JPg3G36jxhu7Mz8FXBGcVwAPwR+GBE91JPb7Mx8OiJqwLTibZsHLX35zcDvm5m/iQj/WyJJO5e2zZHbxPCdiDigmDkcbmwwR2oScKmnNLzvAicCv8zMLZn5S2A36ktZbm3wHLcBPRGxZ0RMAU4beKG4tmBK8fy3gD2BnxQvHxcRe0TEdGBOEctq4NSImFm8Z4+IeBXwPeDoiDiw6H9JRPw2cD+wf0QcUJxz7qCxd4uIFxfNPwO+UxSDM4DHi6LvYOrflEqStK12zpEHFl+KEhFvoD5L+D/Fy0dGxP7FtX3vBNY0+LtKpfMbCGl466gvQ/nSNn0vy8xfDFwcPpLM3BARH6OeBDcAdwIdxcvHU19u+eui/VeZ+bMi16wB/i9wIPClzLwdICLOB24uEs5m4P2Z+b2IeC9wddQvqgc4PzMfiPrmLDdGxC+Kcw58W9oF/HNEbAHuA+YV/TcBCyLibuAH1BOmJEnbaucc+b+BP42IzcAzwDszM4uxb6V+ucRrge8A1zX2zyWVL16YlZZUBUWC6s7MD5QdiyRJVVJmjiwuh/jLzDyx1WNLE8GlnpIkSZLU5pzxkyRJkqQ254yfJEmSJLU5Cz9JkiRJanMWfpIkSZLU5iz8JEmSJKnNWfhJkiRJUpuz8JMkSZKkNvf/AUjdwkSRonC7AAAAAElFTkSuQmCC"/>

위의 상자 그림은 이러한 변수에 많은 이상치가 있음을 확인합니다.


### 변수 분포 확인





이제 히스토그램을 그려서 분포를 확인하고 정규 분포인지 또는 왜곡되어 있는지를 알아보겠습니다. 변수가 정규 분포를 따르면 극단 값 분석을 수행하고, 왜곡되어 있다면 IQR (Interquantile range)를 찾겠습니다.


```python
# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
```

<pre>
Text(0, 0.5, 'RainTomorrow')
</pre>
<img src="data:image/png;base64,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"/>

상기 네 가지 변수는 모두 왜곡되어 있음을 확인할 수 있습니다. 따라서 IQR(Interquantile range)을 사용하여 이상치를 찾겠습니다.


```python
# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
Rainfall outliers are values < -2.4000000000000004 or > 3.2
</pre>
`Rainfall`의 최소값과 최대값은 각각 0.0과 371.0입니다. 따라서 이상치는 값이 3.2보다 큰 값입니다.


```python
# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
Evaporation outliers are values < -11.800000000000002 or > 21.800000000000004
</pre>
`Evaporation`의 최소값과 최대값은 각각 0.0과 145.0입니다. 따라서 이상치는 값이 21.8보다 큰 값입니다.


```python
# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
WindSpeed9am outliers are values < -29.0 or > 55.0
</pre>
`WindSpeed9am`의 최소값과 최대값은 각각 0.0과 130.0입니다. 따라서 이상치는 값이 55.0보다 큰 값입니다.

```python
# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
WindSpeed3pm outliers are values < -20.0 or > 57.0
</pre>
`WindSpeed3pm`의 최소값과 최대값은 각각 0.0과 87.0입니다. 따라서 이상치는 값이 57.0보다 큰 값입니다.


# **8. 피처 벡터 및 타겟 변수 선언** <a class="anchor" id="8"></a>





[Table of Contents](#0.1)



```python
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
```

# **9. 데이터를 학습 및 테스트 세트로 분리** <a class="anchor" id="9"></a>





[Table of Contents](#0.1)



```python
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
# check the shape of X_train and X_test

X_train.shape, X_test.shape
```

<pre>
((113754, 24), (28439, 24))
</pre>
# **10. 피처 엔지니어링** <a class="anchor" id="10"></a>





[Table of Contents](#0.1)





**Feature Engineering** 은 원시 데이터를 유용한 특징으로 변환하여 모델을 이해하고 예측력을 높이는 프로세스입니다. 다른 유형의 변수에 대해 feature engineering을 수행할 것입니다.




먼저 범주형 변수와 수치형 변수를 다시 분리하여 표시하겠습니다.


```python
# check data types in X_train

X_train.dtypes
```

<pre>
Location          object
MinTemp          float64
MaxTemp          float64
Rainfall         float64
Evaporation      float64
Sunshine         float64
WindGustDir       object
WindGustSpeed    float64
WindDir9am        object
WindDir3pm        object
WindSpeed9am     float64
WindSpeed3pm     float64
Humidity9am      float64
Humidity3pm      float64
Pressure9am      float64
Pressure3pm      float64
Cloud9am         float64
Cloud3pm         float64
Temp9am          float64
Temp3pm          float64
RainToday         object
Year               int64
Month              int64
Day                int64
dtype: object
</pre>

```python
# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
```

<pre>
['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
</pre>

```python
# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
```

<pre>
['MinTemp',
 'MaxTemp',
 'Rainfall',
 'Evaporation',
 'Sunshine',
 'WindGustSpeed',
 'WindSpeed9am',
 'WindSpeed3pm',
 'Humidity9am',
 'Humidity3pm',
 'Pressure9am',
 'Pressure3pm',
 'Cloud9am',
 'Cloud3pm',
 'Temp9am',
 'Temp3pm',
 'Year',
 'Month',
 'Day']
</pre>
### 수치형 변수에서 누락된 값을 처리하기





```python
# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```

<pre>
MinTemp            495
MaxTemp            264
Rainfall          1139
Evaporation      48718
Sunshine         54314
WindGustSpeed     7367
WindSpeed9am      1086
WindSpeed3pm      2094
Humidity9am       1449
Humidity3pm       2890
Pressure9am      11212
Pressure3pm      11186
Cloud9am         43137
Cloud3pm         45768
Temp9am            740
Temp3pm           2171
Year                 0
Month                0
Day                  0
dtype: int64
</pre>

```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

<pre>
MinTemp            142
MaxTemp             58
Rainfall           267
Evaporation      12125
Sunshine         13502
WindGustSpeed     1903
WindSpeed9am       262
WindSpeed3pm       536
Humidity9am        325
Humidity3pm        720
Pressure9am       2802
Pressure3pm       2795
Cloud9am         10520
Cloud3pm         11326
Temp9am            164
Temp3pm            555
Year                 0
Month                0
Day                  0
dtype: int64
</pre>

```python
# print percentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

<pre>
MinTemp 0.0044
MaxTemp 0.0023
Rainfall 0.01
Evaporation 0.4283
Sunshine 0.4775
WindGustSpeed 0.0648
WindSpeed9am 0.0095
WindSpeed3pm 0.0184
Humidity9am 0.0127
Humidity3pm 0.0254
Pressure9am 0.0986
Pressure3pm 0.0983
Cloud9am 0.3792
Cloud3pm 0.4023
Temp9am 0.0065
Temp3pm 0.0191
</pre>
### 가정하기





데이터가 완전히 무작위(MCAR)로 누락되었다고 가정합니다. 누락된 값을 대체하는 데 사용할 수 있는 두 가지 방법이 있습니다. 하나는 평균 또는 중앙값 대체이고, 다른 하나는 무작위 표본 대체입니다. 데이터 세트에 이상치가 있는 경우 중앙값 대체를 사용해야 합니다. 따라서 중앙값 대체를 사용할 것입니다. 중앙값 대체는 이상치에 대해 강건합니다.





데이터의 적절한 통계적 측정치로 누락된 값을 채워 넣겠습니다. 이 경우 중앙값을 사용할 것입니다. 대체는 훈련 세트에서 수행되어야 하며, 그런 다음 테스트 세트로 전파되어야 합니다. 즉, 훈련 세트에서 누락된 값을 채우기 위해 사용할 통계 측정치는 훈련 세트에서 추출해야 합니다. 이는 과적합을 피하기 위한 것입니다.


```python
# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)           
      
```


```python
# check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```

<pre>
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustSpeed    0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
Year             0
Month            0
Day              0
dtype: int64
</pre>

```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

<pre>
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustSpeed    0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
Year             0
Month            0
Day              0
dtype: int64
</pre>
현재, 학습 및 테스트 세트의 숫자형 열에는 결측값이 없음을 확인할 수 있습니다.


### 범주형 변수에서 결측값 처리하기



```python
# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()
```

<pre>
Location       0.000000
WindGustDir    0.065114
WindDir9am     0.070134
WindDir3pm     0.026443
RainToday      0.010013
dtype: float64
</pre>

```python
# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
```

<pre>
WindGustDir 0.06511419378659213
WindDir9am 0.07013379749283542
WindDir3pm 0.026443026179299188
RainToday 0.01001283471350458
</pre>

```python
# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
```


```python
# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()
```

<pre>
Location       0
WindGustDir    0
WindDir9am     0
WindDir3pm     0
RainToday      0
dtype: int64
</pre>

```python
# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()
```

<pre>
Location       0
WindGustDir    0
WindDir9am     0
WindDir3pm     0
RainToday      0
dtype: int64
</pre>
마지막으로, X_train 및 X_test에서 결측값이 있는지 확인하겠습니다.



```python
# check missing values in X_train

X_train.isnull().sum()
```

<pre>
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
Year             0
Month            0
Day              0
dtype: int64
</pre>

```python
# check missing values in X_test

X_test.isnull().sum()
```

<pre>
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
Year             0
Month            0
Day              0
dtype: int64
</pre>
X_train 및 X_test에 결측값이 없음을 확인할 수 있습니다.


### 수치형 변수에서 이상치 처리하기




우리는 `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` 열에 이상치가 있음을 확인했습니다. 이상치를 제거하고 위 변수에서 최대값을 상한값으로 대체하기 위해 상위 코딩(top-coding) 방법을 사용하겠습니다.


```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
```


```python
X_train.Rainfall.max(), X_test.Rainfall.max()
```

<pre>
(3.2, 3.2)
</pre>

```python
X_train.Evaporation.max(), X_test.Evaporation.max()
```

<pre>
(21.8, 21.8)
</pre>

```python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```

<pre>
(55.0, 55.0)
</pre>

```python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```

<pre>
(57.0, 57.0)
</pre>

```python
X_train[numerical].describe()
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>1017.640649</td>
      <td>1015.241101</td>
      <td>4.651801</td>
      <td>4.703588</td>
      <td>16.995062</td>
      <td>21.688643</td>
      <td>2012.759727</td>
      <td>6.404021</td>
      <td>15.710419</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>6.738680</td>
      <td>6.675168</td>
      <td>2.292726</td>
      <td>2.117847</td>
      <td>6.463772</td>
      <td>6.855649</td>
      <td>2.540419</td>
      <td>3.427798</td>
      <td>8.796821</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>980.500000</td>
      <td>977.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-7.200000</td>
      <td>-5.400000</td>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>1013.500000</td>
      <td>1011.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>12.300000</td>
      <td>16.700000</td>
      <td>2011.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>1017.600000</td>
      <td>1015.200000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>16.700000</td>
      <td>21.100000</td>
      <td>2013.000000</td>
      <td>6.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>1021.800000</td>
      <td>1019.400000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>21.500000</td>
      <td>26.300000</td>
      <td>2015.000000</td>
      <td>9.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1041.000000</td>
      <td>1039.600000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>40.200000</td>
      <td>46.700000</td>
      <td>2017.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
    </tr>
  </tbody>
</table>
</div>


`Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` 열의 이상치가 제거되었고, 상한값으로 대체되었음을 확인할 수 있습니다.

### 범주형 변수 인코딩하기


```python
categorical
```

<pre>
['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
</pre>

```python
X_train[categorical].head()
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
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>Witchcliffe</td>
      <td>S</td>
      <td>SSE</td>
      <td>S</td>
      <td>No</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>Cairns</td>
      <td>ENE</td>
      <td>SSE</td>
      <td>SE</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>AliceSprings</td>
      <td>E</td>
      <td>NE</td>
      <td>N</td>
      <td>No</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>Cairns</td>
      <td>ESE</td>
      <td>SSE</td>
      <td>E</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>Newcastle</td>
      <td>W</td>
      <td>N</td>
      <td>SE</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



```python
# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```


```python
X_train.head()
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
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday_0</th>
      <th>RainToday_1</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>Witchcliffe</td>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>S</td>
      <td>41.0</td>
      <td>SSE</td>
      <td>S</td>
      <td>...</td>
      <td>1013.4</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>18.8</td>
      <td>20.4</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>4</td>
      <td>25</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>Cairns</td>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>ENE</td>
      <td>33.0</td>
      <td>SSE</td>
      <td>SE</td>
      <td>...</td>
      <td>1013.1</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>26.4</td>
      <td>27.5</td>
      <td>1</td>
      <td>0</td>
      <td>2015</td>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>AliceSprings</td>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>E</td>
      <td>31.0</td>
      <td>NE</td>
      <td>N</td>
      <td>...</td>
      <td>1013.6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.5</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>10</td>
      <td>19</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>Cairns</td>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>ESE</td>
      <td>37.0</td>
      <td>SSE</td>
      <td>E</td>
      <td>...</td>
      <td>1010.8</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>27.3</td>
      <td>29.4</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>10</td>
      <td>30</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>Newcastle</td>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>W</td>
      <td>39.0</td>
      <td>N</td>
      <td>SE</td>
      <td>...</td>
      <td>1015.2</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>22.2</td>
      <td>27.0</td>
      <td>0</td>
      <td>1</td>
      <td>2012</td>
      <td>11</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>


WRainToday 변수에서 RainToday_0 및 RainToday_1 두 가지 추가 변수가 생성된 것을 확인할 수 있습니다.


이제 X_train 학습 세트를 만들겠습니다.



```python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
```


```python
X_train.head()
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>28.0</td>
      <td>65.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>33.0</td>
      <td>7.0</td>
      <td>19.0</td>
      <td>71.0</td>
      <td>59.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>31.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>2.0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>37.0</td>
      <td>22.0</td>
      <td>19.0</td>
      <td>59.0</td>
      <td>53.0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>72.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>


마찬가지로, X_test 테스트 세트를 만들겠습니다.


```python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
```


```python
X_test.head()
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86232</th>
      <td>17.4</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>3.6</td>
      <td>11.1</td>
      <td>33.0</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>63.0</td>
      <td>61.0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>57576</th>
      <td>6.8</td>
      <td>14.4</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>8.5</td>
      <td>46.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>80.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124071</th>
      <td>10.1</td>
      <td>15.4</td>
      <td>3.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>31.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>70.0</td>
      <td>61.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117955</th>
      <td>14.4</td>
      <td>33.4</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>11.6</td>
      <td>41.0</td>
      <td>9.0</td>
      <td>17.0</td>
      <td>40.0</td>
      <td>23.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>133468</th>
      <td>6.8</td>
      <td>14.3</td>
      <td>3.2</td>
      <td>0.2</td>
      <td>7.3</td>
      <td>28.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>92.0</td>
      <td>47.0</td>
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
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>


이제 모델 구축을 위해 모든 피처 변수를 동일한 척도로 매핑해야 합니다. 이를 feature scaling 이라고 합니다. 다음과 같이 수행하겠습니다.


# **11. 피처 스케일링** <a class="anchor" id="11"></a>





[Table of Contents](#0.1)



```python
X_train.describe()
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>



```python
cols = X_train.columns
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```


```python
X_train = pd.DataFrame(X_train, columns=[cols])
```


```python
X_test = pd.DataFrame(X_test, columns=[cols])
```


```python
X_train.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.484406</td>
      <td>0.530004</td>
      <td>0.210962</td>
      <td>0.236312</td>
      <td>0.554562</td>
      <td>0.262667</td>
      <td>0.254148</td>
      <td>0.326575</td>
      <td>0.688675</td>
      <td>0.515095</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.151741</td>
      <td>0.134105</td>
      <td>0.369949</td>
      <td>0.129528</td>
      <td>0.190999</td>
      <td>0.101682</td>
      <td>0.160119</td>
      <td>0.152384</td>
      <td>0.189356</td>
      <td>0.205307</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.375297</td>
      <td>0.431002</td>
      <td>0.000000</td>
      <td>0.183486</td>
      <td>0.565517</td>
      <td>0.193798</td>
      <td>0.127273</td>
      <td>0.228070</td>
      <td>0.570000</td>
      <td>0.370000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.479810</td>
      <td>0.517958</td>
      <td>0.000000</td>
      <td>0.220183</td>
      <td>0.586207</td>
      <td>0.255814</td>
      <td>0.236364</td>
      <td>0.333333</td>
      <td>0.700000</td>
      <td>0.520000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.593824</td>
      <td>0.623819</td>
      <td>0.187500</td>
      <td>0.247706</td>
      <td>0.600000</td>
      <td>0.310078</td>
      <td>0.345455</td>
      <td>0.421053</td>
      <td>0.830000</td>
      <td>0.650000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>


이제 Logistic Regression 분류기에 X_train 데이터셋을 입력할 준비가 되었습니다. 다음과 같이 수행하겠습니다.


# **12. 모델 학습** <a class="anchor" id="12"></a>





[Table of Contents](#0.1)



```python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)
```

<pre>
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
</pre>
# **13. 결과 예측** <a class="anchor" id="13"></a>





[Table of Contents](#0.1)



```python
y_pred_test = logreg.predict(X_test)

y_pred_test
```

<pre>
array(['No', 'No', 'No', ..., 'No', 'No', 'Yes'], dtype=object)
</pre>
### predict_proba 메소드





**predict_proba** 메소드는 대상 변수(여기서는 0과 1)에 대한 확률을 배열 형태로 제공합니다.



`0 은 비가 오지 않을 확률을` , `1 은 비가 올 확률을 나타냅니다.`



```python
# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0]
```

<pre>
array([0.91382428, 0.83565645, 0.82033915, ..., 0.97674285, 0.79855098,
       0.30734161])
</pre>

```python
# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]
```

<pre>
array([0.08617572, 0.16434355, 0.17966085, ..., 0.02325715, 0.20144902,
       0.69265839])
</pre>
# **14. 정확도 점수 확인** <a class="anchor" id="14"></a>





[Table of Contents](#0.1)



```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

<pre>
Model accuracy score: 0.8502
</pre>
여기서, y_test는 테스트 세트의 실제 클래스 레이블이고, y_pred_test는 예측된 클래스 레이블입니다.


### 학습 세트와 테스트 세트의 정확도 비교하기





이제 학습 세트와 테스트 세트의 정확도를 비교하여 과적합 여부를 확인하겠습니다.



```python
y_pred_train = logreg.predict(X_train)

y_pred_train
```

<pre>
array(['No', 'No', 'No', ..., 'No', 'No', 'No'], dtype=object)
</pre>

```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

<pre>
Training-set accuracy score: 0.8476
</pre>
### 과적합과 과소적합 확인하기



```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

<pre>
Training set score: 0.8476
Test set score: 0.8502
</pre>
학습 세트의 정확도 점수는 0.8476이고, 테스트 세트의 정확도는 0.8501입니다. 이 두 값은 상당히 유사합니다. 따라서 과적합의 문제는 없습니다.



Logistic Regression에서 C의 기본값은 1입니다. 이 값은 학습 세트와 테스트 세트에서 모두 약 85% 정확도의 좋은 성능을 제공합니다. 하지만 학습 세트와 테스트 세트 모두에서 모델의 성능이 매우 유사합니다. 이는 과소적합의 가능성이 높은 상황입니다.



따라서 C 값을 늘리고 더 유연한 모델을 적합시키겠습니다.



```python
# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)
```

<pre>
LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
</pre>

```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

<pre>
Training set score: 0.8478
Test set score: 0.8505
</pre>
우리는 C=100일 때 더 높은 테스트 세트 정확도와 약간 높은 훈련 세트 정확도를 확인할 수 있습니다. 따라서 더 복잡한 모델이 더 좋은 성능을 발휘한다는 결론을 내릴 수 있습니다.


이제 우리는 기본값인 C=1 대신 C=0.01과 같이 더 규제된 모델을 사용할 때 무엇이 일어나는지 조사해 볼 것입니다.


```python
# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)
```

<pre>
LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
</pre>

```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

<pre>
Training set score: 0.8409
Test set score: 0.8448
</pre>
따라서 C=0.01과 같이 더 규제된 모델을 사용하면 기본값에 비해 훈련 세트와 테스트 세트의 정확도가 모두 낮아집니다.

### 모델 정확도와 null 정확도를 비교





모델 정확도와 null 정확도를 비교하면, 모델 정확도는 0.8501입니다. 그러나 이는 위의 정확도만으로 우리 모델이 매우 좋다고 말할 수는 없습니다. null 정확도와 비교해야 합니다. null 정확도는 항상 가장 빈번한 클래스를 예측하는 것으로 얻을 수 있는 정확도입니다.

따라서 먼저 테스트 세트에서 클래스 분포를 확인해야 합니다.


```python
# check class distribution in test set

y_test.value_counts()
```

<pre>
No     22067
Yes     6372
Name: RainTomorrow, dtype: int64
</pre>
가장 빈번한 클래스의 발생 횟수는 22067입니다. 그러므로 22067을 전체 발생 횟수로 나누어 null 정확도를 계산할 수 있습니다.



```python
# check null accuracy score

null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

<pre>
Null accuracy score: 0.7759
</pre>
모델 정확도 점수는 0.8501이지만 null 정확도 점수는 0.7759입니다. 그러므로 우리는 로지스틱 회귀 모델이 클래스 레이블을 예측하는 데 아주 잘 작동한다고 결론을 내릴 수 있습니다.


위 분석을 기반으로, 분류 모델의 정확도가 매우 좋다는 결론을 내릴 수 있습니다. 우리 모델은 클래스 레이블을 예측하는 데 아주 잘 작동하고 있습니다.





하지만, 모델은 값의 분포를 알려주지 않습니다. 또한, 분류기가 만드는 오류의 유형에 대해 아무것도 알려주지 않습니다. 





이런 경우, 우리는 혼동 행렬이라는 도구를 사용할 수 있습니다.


# **15. 혼동 행렬** <a class="anchor" id="15"></a>





[Table of Contents](#0.1)





혼동 행렬(Confusion Matrix)은 분류 알고리즘의 성능을 요약하는 도구입니다. 혼동 행렬은 모델의 성능과 모델이 만드는 에러의 유형에 대한 명확한 그림을 제공합니다. 이는 각 범주별로 올바른 및 잘못된 예측의 요약을 제공하며, 표 형태로 표시됩니다.





분류 모델의 성능을 평가할 때는 네 가지 결과가 가능합니다. 이 네 가지 결과는 아래와 같이 설명됩니다.





**True Positives (TP)** – 특정 클래스에 속하는 관측치를 예측하고 그 관측치가 실제로 그 클래스에 속한 경우 발생합니다.





**True Negatives (TN)** – 특정 클래스에 속하지 않는 관측치를 예측하고 그 관측치가 실제로 그 클래스에 속하지 않는 경우 발생합니다.





**False Positives (FP)** –특정 클래스에 속하는 관측치를 예측하지만, 그 관측치가 실제로는 그 클래스에 속하지 않는 경우 발생합니다. 이러한 유형의 에러는 **제 1 종 오류(Type I error)** 라고 합니다.







**False Negatives (FN)** – 특정 클래스에 속하지 않는 관측치를 예측하지만, 그 관측치가 실제로는 그 클래스에 속하는 경우 발생합니다. 이러한 유형의 에러는 매우 심각한 오류이며 **제 2 종 오류(Type II error)** 라고 합니다.







이 네 가지 결과는 아래의 혼동 행렬에서 요약됩니다.




```python
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
```

<pre>
Confusion matrix

 [[20892  1175]
 [ 3086  3286]]

True Positives(TP) =  20892

True Negatives(TN) =  3286

False Positives(FP) =  1175

False Negatives(FN) =  3086
</pre>
혼동 행렬은 `20892 + 3285 = 24177 올바른 예측` 과 `3087 + 1175 = 4262 잘못된 예측`을 보여줍니다.





따라서 이 경우에는 다음과 같습니다.





- `True Positives` (실제 Positive:1 and 예측 Positive:1) - 20892





- `True Negatives` (실제 Negative:0 and 예측 Negative:0) - 3285





- `False Positives` (실제 Negative:0 but 예측 Positive:1) - 1175 `(Type I error)`





- `False Negatives` (실제 Positive:1 but 예측 Negative:0) - 3087 `(Type II error)`



```python
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```

<pre>
<matplotlib.axes._subplots.AxesSubplot at 0x7f28b1306208>
</pre>
<img src="data:image/png;base64,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"/>

# **16. 분류 메트릭스** <a class="anchor" id="16"></a>





[Table of Contents](#0.1)


## 분류 보고서





**분류 보고서** 는 분류 모델 성능을 평가하는 다른 방법입니다. 모델의 정밀도(precision), 재현율(recall), f1 스코어 및 지원 데이터 개수를 나타냅니다.



아래에서 이에 대해 설명합니다.



```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

<pre>
              precision    recall  f1-score   support

          No       0.87      0.95      0.91     22067
         Yes       0.74      0.52      0.61      6372

    accuracy                           0.85     28439
   macro avg       0.80      0.73      0.76     28439
weighted avg       0.84      0.85      0.84     28439

</pre>
## 분류 정확도


```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
```


```python
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
```

<pre>
Classification accuracy : 0.8502
</pre>
## 분류 오류



```python
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
```

<pre>
Classification error : 0.1498
</pre>
## 정밀도





**정밀도**는 모든 예측된 양성 예측(outcomes) 중 올바르게 예측된 양성 예측의 비율입니다. 이는 실제 양성(True Positives, TP)과 거짓 양성(False Positives, FP)의 합(TP + FP)으로 나누어 구할 수 있습니다.





**정밀도**는 양성 클래스에 대한 정확한 예측 비율을 나타냅니다. 즉, 양성으로 예측한 데이터 중 실제로 양성인 데이터의 비율을 의미합니다. 이는 음성 클래스보다 양성 클래스에 더 집중합니다.






수식적으로, 정밀도는 TP / (TP + FP)로 정의됩니다.








```python
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
```

<pre>
Precision : 0.9468
</pre>
## 재현율





재현율은 모든 실제 양성(outcomes) 중 올바르게 예측된 양성 예측의 비율입니다. 이는 실제 양성(TP)과 거짓 음성(False Negatives, FN)의 합(TP + FN)으로 나누어 구할 수 있습니다.

**재현율**은 **민감도**라고도 불립니다.





**재현율** 은 실제 양성 클래스에 속한 샘플들 중 모델이 정확하게 양성으로 예측한 비율을 의미합니다.




수식적으로, 재현율은 TP / (TP + FN)으로 정의됩니다.










```python
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
```

<pre>
Recall or Sensitivity : 0.8713
</pre>
## 진 양성율





**진양성율**은 **재현율**과 동일한 개념입니다.




```python
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```

<pre>
True Positive Rate : 0.8713
</pre>
## 가 양성율
 


```python
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```

<pre>
False Positive Rate : 0.2634
</pre>
## 특이도



```python
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
```

<pre>
Specificity : 0.7366
</pre>
## f1-스코어





**f1-스코어** 는 정밀도와 재현율의 가중 조화 평균입니다. 가장 좋은 f1 스코어는 1.0이며, 가장 나쁜 경우는 0.0입니다. 

**f1-스코어**는 정밀도와 재현율의 가중 조화 평균으로 계산됩니다. 따라서 **f1-스코어**는 정확도 측정보다 항상 낮습니다. 분류기 모델을 비교하기 위해서는 가중 평균(f1 스코어)을 사용해야 합니다..





## 지원 데이터





**지원 데이터**는 데이터셋에서 해당 클래스가 발생한 실제 빈도수를 의미합니다.

# **17. 임계값 조정** <a class="anchor" id="17"></a>





[Table of Contents](#0.1)



```python
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```

<pre>
array([[0.91382428, 0.08617572],
       [0.83565645, 0.16434355],
       [0.82033915, 0.17966085],
       [0.99025322, 0.00974678],
       [0.95726711, 0.04273289],
       [0.97993908, 0.02006092],
       [0.17833011, 0.82166989],
       [0.23480918, 0.76519082],
       [0.90048436, 0.09951564],
       [0.85485267, 0.14514733]])
</pre>
### 예측





- 관측치들은 각 행에서 숫자들의 합이 1이 되도록 나타납니다.





- 2개의 열은 0과 1이라는 2개의 클래스를 나타냅니다.



    - 클래스 0 - 내일 비가 오지 않을 확률    

    

    - 클래스 1 - 내일 비가 올 확률

        

    

- 예측 확률의 중요성



    - 확률에 따라 관측값을 순위별로 정렬할 수 있다.





- predict_proba 과정


    - 확률을 예측한다.

    

    - 가장 높은 확률을 가진 클래스를 선택한다.    

    

    

- 분류 임계치



    - 분류 임계치는 0.5이다.    

    

    - 클래스 1 - 확률이 0.5보다 크면 내일 비가 올 확률로 예측된다.  

    

    - 클래스 0 - 확률이 0.5보다 작으면 내일 비가 오지 않을 확률로 예측된다.


    




```python
# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
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
      <th>Prob of - No rain tomorrow (0)</th>
      <th>Prob of - Rain tomorrow (1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.913824</td>
      <td>0.086176</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.835656</td>
      <td>0.164344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.820339</td>
      <td>0.179661</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.990253</td>
      <td>0.009747</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.957267</td>
      <td>0.042733</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.979939</td>
      <td>0.020061</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.178330</td>
      <td>0.821670</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.234809</td>
      <td>0.765191</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.900484</td>
      <td>0.099516</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.854853</td>
      <td>0.145147</td>
    </tr>
  </tbody>
</table>
</div>



```python
# print the first 10 predicted probabilities for class 1 - Probability of rain

logreg.predict_proba(X_test)[0:10, 1]
```

<pre>
array([0.08617572, 0.16434355, 0.17966085, 0.00974678, 0.04273289,
       0.02006092, 0.82166989, 0.76519082, 0.09951564, 0.14514733])
</pre>

```python
# store the predicted probabilities for class 1 - Probability of rain

y_pred1 = logreg.predict_proba(X_test)[:, 1]
```


```python
# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
```

<pre>
Text(0, 0.5, 'Frequency')
</pre>
<img src="data:image/png;base64,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"/>

### 관측값





- 우리는 위의 히스토그램이 매우 양의 왜곡을 가지고 있다는 것을 볼 수 있다.





- 첫 번째 열은 확률이 0.0에서 0.1 사이인 관측값이 약 15000개 정도 있다는 것을 알려준다.





- 확률이 0.5보다 큰 일부 관측값들이 있다.





- 따라서 이러한 일부 관측값들은 내일 비가 올 것으로 예측되고 있다.





- 대다수의 관측값들은 내일 비가 오지 않을 것으로 예측되고 있다.


### 임계치 낮추기


```python
from sklearn.preprocessing import binarize

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = binarize(y_pred1, i/10)
    
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
    cm1 = confusion_matrix(y_test, y_pred2)
        
    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')
```

<pre>
With 0.1 threshold the Confusion Matrix is  

 [[12726  9341]
 [  547  5825]] 

 with 18551 correct predictions,  

 9341 Type I errors( False Positives),  

 547 Type II errors( False Negatives),  

 Accuracy score:  0.6523084496641935 

 Sensitivity:  0.9141556811048337 

 Specificity:  0.5766982371867494 

 ==================================================== 


With 0.2 threshold the Confusion Matrix is  

 [[17066  5001]
 [ 1234  5138]] 

 with 22204 correct predictions,  

 5001 Type I errors( False Positives),  

 1234 Type II errors( False Negatives),  

 Accuracy score:  0.7807588171173389 

 Sensitivity:  0.8063402385436284 

 Specificity:  0.7733720034440568 

 ==================================================== 


With 0.3 threshold the Confusion Matrix is  

 [[19080  2987]
 [ 1872  4500]] 

 with 23580 correct predictions,  

 2987 Type I errors( False Positives),  

 1872 Type II errors( False Negatives),  

 Accuracy score:  0.8291430781673055 

 Sensitivity:  0.7062146892655368 

 Specificity:  0.8646395069560883 

 ==================================================== 


With 0.4 threshold the Confusion Matrix is  

 [[20191  1876]
 [ 2517  3855]] 

 with 24046 correct predictions,  

 1876 Type I errors( False Positives),  

 2517 Type II errors( False Negatives),  

 Accuracy score:  0.845529027040332 

 Sensitivity:  0.6049905838041432 

 Specificity:  0.9149861784565188 

 ==================================================== 


</pre>
### 코멘트





- 바이너리 문제에서는 예측된 확률을 클래스 예측으로 변환하기 위해 기본적으로 0.5의 임계치를 사용한다.




- 임계치를 조정하여 민감도 또는 특이도를 증가시킬 수 있다.





- 민감도와 특이도는 서로 역의 관계를 가진다. 한쪽이 증가하면 다른 쪽은 감소하고 그 반대도 마찬가지이다.





- 임계치 레벨을 높이면 정확도가 증가하는 것을 볼 수 있다.




- 임계치 레벨 조정은 모델 구축 프로세스에서 마지막 단계 중 하나여야 한다.

# **18. ROC - AUC** <a class="anchor" id="18"></a>





[Table of Contents](#0.1)







## ROC 곡선





분류 모델의 성능을 시각적으로 측정하는 또 다른 도구로 **ROC 곡선**이 있습니다. **ROC 곡선**은 Receiver Operating Characteristic Curve의 약어로, 분류 모델의 다양한 분류 임계값에서의 성능을 보여주는 그래프입니다.







**ROC 곡선**은 다양한 임계값에서의 **True Positive Rate (TPR)** 와 **False Positive Rate (FPR)**를  나타냅니다.







**True Positive Rate (TPR)**은 **Recall**이라고도 불리며, TP(True Positive)를 (TP + FN)으로 나눈 비율입니다.







**False Positive Rate (FPR)**는 FP(False Positive)를 (FP + TN)으로 나눈 비율입니다.









ROC 곡선에서는 TPR과 FPR의 단일 지점에 초점을 맞춥니다. 이를 통해 다양한 분류 임계값에서의 TPR과 FPR로 구성된 ROC 곡선의 일반적인 성능을 파악할 수 있습니다. 따라서 ROC 곡선은 다양한 분류 임계값에서의 TPR 대 FPR을 그래프로 나타낸 것입니다. 임계값을 낮출수록 양성으로 분류되는 항목이 늘어날 수 있습니다. 이는 TP와 FP를 모두 증가시킵니다.






```python
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
```

<img src="data:image/png;base64,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"/>

ROC 곡선을 통해 특정 맥락에서 민감도와 특이도를 균형있게 조절할 수 있는 분류 임계값을 선택할 수 있습니다.

## ROC-AUC





**ROC AUC** 는 **Receiver Operating Characteristic - Area Under Curve**의 약어로, 분류기 성능을 비교하는 기술입니다. 이 기술에서는 곡선 아래 면적(AUC)을 측정합니다. 완벽한 분류기는 ROC AUC가 1이 되고, 완전히 무작위인 분류기는 ROC AUC가 0.5가 됩니다.




따라서 **ROC AUC** 는 곡선 아래 면적의 백분율입니다.



```python
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```

<pre>
ROC AUC : 0.8729
</pre>
### 코멘트





- ROC AUC는 분류기의 성능을 요약한 단일 숫자입니다. 값이 높을수록 분류기의 성능이 더 좋습니다.


- 의사결정나무 모델의 ROC AUC가 1에 가깝기 때문에, 이 모델은 내일 비가 올 확률을 잘 예측한다고 결론을 내릴 수 있습니다.


```python
# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
```

<pre>
Cross validated ROC AUC : 0.8695
</pre>
# **19. k-Fold 교차 검증** <a class="anchor" id="19"></a>





[Table of Contents](#0.1)



```python
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
```

<pre>
Cross-validation scores:[0.84686387 0.84624852 0.84633642 0.84963298 0.84773626]
</pre>
우리는 교차 검증 정확도를 평균을 내어 요약할 수 있습니다.



```python
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

<pre>
Average cross-validation score: 0.8474
</pre>
원래 모델 점수는 0.8476이고, 교차 검증의 평균 점수는 0.8474입니다. 따라서 교차 검증이 성능 향상을 가져오지 않는 것으로 결론 을 수 있습니다.


# **20. 그리드 서치를 사용한 하이퍼파라미터 최적화** <a class="anchor" id="20"></a>





[Table of Contents](#0.1)



```python
from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)
```

<pre>
GridSearchCV(cv=5, error_score='raise-deprecating',
             estimator=LogisticRegression(C=1.0, class_weight=None, dual=False,
                                          fit_intercept=True,
                                          intercept_scaling=1, l1_ratio=None,
                                          max_iter=100, multi_class='warn',
                                          n_jobs=None, penalty='l2',
                                          random_state=0, solver='liblinear',
                                          tol=0.0001, verbose=0,
                                          warm_start=False),
             iid='warn', n_jobs=None,
             param_grid=[{'penalty': ['l1', 'l2']}, {'C': [1, 10, 100, 1000]}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='accuracy', verbose=0)
</pre>

```python
# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
```

<pre>
GridSearch CV best score : 0.8474


Parameters that give the best results : 

 {'penalty': 'l1'}


Estimator that was chosen by the search : 

 LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l1',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
</pre>

```python
# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
```

<pre>
GridSearch CV score on test set: 0.8507
</pre>
### 코멘트





- 원래 모델의 테스트 정확도는 0.8501이며, GridSearch CV 정확도는 0.8507입니다. 




- 이 경우 GridSearch CV가 모델의 성능을 개선시켰음을 확인할 수 있습니다.


# **21. 결과 및 결론** <a class="anchor" id="21"></a>





[Table of Contents](#0.1)


1.	로지스틱 회귀 모델 정확도 점수는 0.8501입니다. 따라서 이 모델은 내일 오스트레일리아에서 비가 올지 말지 예측하는 데 매우 잘 작동합니다.


2.	적은 수의 관측치가 내일 비가 올 것이라고 예측하고 있습니다. 대다수의 관측치는 내일 비가 오지 않을 것으로 예측하고 있습니다.



3.	모델은 과적합의 증거가 없습니다.



4.	C 값의 증가는 높은 테스트 세트 정확도와 약간 증가한 훈련 세트 정확도를 가져옵니다. 따라서 보다 복잡한 모델이 더 나은 성능을 발휘할 것으로 결론짓을 수 있습니다.



5.	임계값의 증가는 정확도를 높이게 됩니다.



6.	우리 모델의 ROC AUC는 1에 가까워지고 있습니다. 따라서 우리의 분류기가 내일 비가 올지 말지 예측하는 데 좋은 성능을 발휘하고 있음을 결론짓을 수 있습니다.


7.	원래 모델 정확도 점수는 0.8501이고, RFECV 이후 정확도 점수는 0.8500입니다. 따라서 우리는 feature set을 줄이면서도 거의 동일한 정확도를 얻을 수 있습니다.



8.	원래 모델에서 FP는 1175이며, FP1은 1174입니다. 따라서 거의 동일한 수의 false positive를 얻을 수 있습니다. 또한, FN은 3087이고, FN1은 3091입니다. 따라서 false negative는 약간 높아졌습니다.



9.	우리의 원래 모델 점수는 0.8476입니다. 평균 교차 검증 점수는 0.8474입니다. 따라서 교차 검증은 성능 향상을 가져오지 않는 것으로 결론을 내릴 수 있습니다.



10.	원래 모델의 테스트 정확도는 0.8501이고, GridSearch CV 정확도는 0.8507입니다. 이 경우 GridSearch CV가 성능을 향상시키는 것을 확인할 수 있습니다.



# **22. 참고 문헌** <a class="anchor" id="22"></a>





[Table of Contents](#0.1)







이 프로젝트는 아래의 책과 웹 사이트에서 영감을 받아 제작되었습니다.





1. Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron



2. Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido



3. Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves



4. Udemy course – Feature Engineering for Machine Learning by Soledad Galli



5. Udemy course – Feature Selection for Machine Learning by Soledad Galli



6. https://en.wikipedia.org/wiki/Logistic_regression



7. https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html



8. https://en.wikipedia.org/wiki/Sigmoid_function



9. https://www.statisticssolutions.com/assumptions-of-logistic-regression/



10. https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python



11. https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression



12. https://www.ritchieng.com/machine-learning-evaluate-classification-model/



[Go to Top](#0)

