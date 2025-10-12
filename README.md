# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
 import pandas as pd
 import numpy as np
 import seaborn as sns
 from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import accuracy_score, confusion_matrix
 data=pd.read_csv("income.csv",na_values=[ " ?"])
 data
```
<img width="907" height="261" alt="image" src="https://github.com/user-attachments/assets/42421e90-1db4-459f-94f5-ab7b0b72a62d" />

```
data.isnull().sum()
```
<img width="241" height="317" alt="image" src="https://github.com/user-attachments/assets/22dd6c9d-b756-4c23-88c2-e821d0f293a2" />

```
 missing=data[data.isnull().any(axis=1)]
 missing
```
<img width="1772" height="727" alt="image" src="https://github.com/user-attachments/assets/75045767-558a-4e75-941c-1a12dcce7699" />


```
 data2=data.dropna(axis=0)
 data2
```
<img width="1629" height="477" alt="image" src="https://github.com/user-attachments/assets/27426cd4-0b4f-47f2-ae52-51e72a6b10a1" />


```
 sal=data["SalStat"]
 data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])
```
<img width="1422" height="376" alt="image" src="https://github.com/user-attachments/assets/3e50d230-3bef-4f05-88ee-5494f7d53dc0" />


```
  sal2=data2['SalStat']
 dfs=pd.concat([sal,sal2],axis=1)
 dfs
```
<img width="525" height="459" alt="image" src="https://github.com/user-attachments/assets/0bb4f585-a6c7-4c89-9110-e67af8fe7821" />


```
  data2
```
<img width="1539" height="475" alt="image" src="https://github.com/user-attachments/assets/321d539d-0115-40e9-aec9-e4cc02f1c36c" />



```
   new_data=pd.get_dummies(data2, drop_first=True)
 new_data
```
<img width="1850" height="271" alt="image" src="https://github.com/user-attachments/assets/af86e223-4f9e-491d-b0b3-04d102bc8cdd" />



```
 columns_list=list(new_data.columns)
 print(columns_list)
```

<img width="1856" height="66" alt="image" src="https://github.com/user-attachments/assets/f9b20bcc-d479-461a-a8dd-02febdc388e6" />


```
 features=list(set(columns_list)-set(['SalStat']))
 print(features)
```
<img width="1861" height="64" alt="image" src="https://github.com/user-attachments/assets/8c3f7f4e-b58f-4deb-84a5-b70a1f697d93" />



```
 y=new_data['SalStat'].values
 print(y)
```                                                                         
<img width="329" height="85" alt="image" src="https://github.com/user-attachments/assets/855b7842-7c03-42ab-8ad0-ecf33eec7dbb" />





```
x=new_data[features].values
print(x)
```
<img width="477" height="133" alt="image" src="https://github.com/user-attachments/assets/3f6b4f2b-3498-42bd-94c0-8d68022dfa3b" />


```
 train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
 KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
 KNN_classifier.fit(train_x,train_y)
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)
```
<img width="207" height="46" alt="image" src="https://github.com/user-attachments/assets/f905c727-23aa-4b62-a2fd-ceb782cb420e" />





```
 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)
```

<img width="234" height="32" alt="image" src="https://github.com/user-attachments/assets/76c1fd4d-ff52-4eb4-b0ed-28e1d18f9d93" />





```
  print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="288" height="36" alt="image" src="https://github.com/user-attachments/assets/78ce0b87-99ec-405f-a8f9-00eab170f1aa" />




```
  data.shape
```
<img width="139" height="25" alt="image" src="https://github.com/user-attachments/assets/85bb71a1-81db-4640-9838-2fc7525abd33" />



```  
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
 data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 selected_feature_indices=selector.get_support(indices=True)
 selected_features=x.columns[selected_feature_indices]
 print("Selected Features:")
 print(selected_features)
```
<img width="1706" height="81" alt="image" src="https://github.com/user-attachments/assets/54229f53-56d8-4e7e-8a50-efad997b1e60" />



```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```
<img width="543" height="190" alt="image" src="https://github.com/user-attachments/assets/87f506f3-3c95-481b-8020-e100a478ee40" />


 ```
tips.time.unique()
```
<img width="387" height="54" alt="image" src="https://github.com/user-attachments/assets/e4013577-2d95-40b9-aee9-0ce12837410d" />


```
 contingency_table=pd.crosstab(tips['sex'],tips['time'])
 print(contingency_table)
```
<img width="239" height="84" alt="image" src="https://github.com/user-attachments/assets/0ae5b384-b875-48b3-a2ca-5a5e90a023a8" />



 ``` 
 chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")
```
<img width="380" height="45" alt="image" src="https://github.com/user-attachments/assets/b2f7c8a4-42c1-4ec1-ba69-ab99fa40b8ce" />






































































# RESULT:
       
