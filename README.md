# Ex-07-Feature-Selection
### AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

### Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

### ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file
# CODE
```
Developed By:DIVYA.K
Register no:212222230035
```
```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/titanic_dataset.csv")

df.columns
df.shape
x=df.drop("Survived",1)
y=df['Survived']
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df1.columns
df1['Age'].isnull().sum()
df1['Age'].fillna(method='ffill')
df1['Age']=df1['Age'].fillna(method='ffill')
df1['Age'].isnull().sum()
feature=SelectKBest(mutual_info_classif,k=3)

df1.columns
cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]

df1.columns
x=df1.iloc[:,0:6]
y=df1.iloc[:,6]

x.columns
y=y.to_frame()

y.columns
from sklearn.feature_selection import SelectKBest

data=pd.read_csv("/content/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']

x
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data[ "Embarked" ]=data ["Embarked"] .astype ("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data[ "Embarked" ]=data ["Embarked"] .cat.codes

data
k=5
selector = SelectKBest(score_func=chi2,k=k)
x_new = selector.fit_transform(x,y)

selected_feature_indices = selector.get_support(indices=True)

selected_feature_indices = selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features: ")
print(selected_features)
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

sfm = SelectFromModel(model, threshold='mean')

sfm.fit(x,y)

selected_feature = x.columns[sfm.get_support()]

print("Selected Features:")
print(selected_feature)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()

num_features_to_remove =2
rfe = RFE(model, n_features_to_select=(len(x.columns) - num_features_to_remove))

rfe.fit(x,y)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()

num_features_to_remove =2
rfe = RFE(model, n_features_to_select=(len(x.columns) - num_features_to_remove))

rfe.fit(x,y)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x,y)

feature_importances = model.feature_importances_

threshold = 0.15

selected_features = x.columns[feature_importances > threshold]

print("Selected Features:")
print(selected_feature)
```

# OUPUT

![282469823-b4566409-e2dd-499e-82a5-39d8daff155e](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/8997c63f-6701-4f31-94f8-68c3af9a22d6)
![282469925-be73f7e1-a04d-4eb6-895f-bf64baeb568a](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/5d4ce9aa-5dcc-4b21-894c-4d59d856d81c)

![282469985-6aa71ac3-e5b7-48c3-96ea-21ab36b116ed](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/d72761fd-f4b6-4b2c-803b-acc0557aacab)
![282470066-d35e1f77-471e-4b22-82a4-ffed00ac3ea0](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/99d17eef-d865-4163-8e53-e3940b3dd385)
![282470560-ed42a4da-be8f-4bd8-9f22-d2e8ec549d3a](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/13aa4a98-ee92-4cf4-b362-5b110535b768)

![282470666-9d969fc5-f727-4e35-b627-495e4c772655](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/4fc0357b-e89d-4d29-a596-b5bb52deca32)

![282471087-90193309-c005-4549-9823-b7598c5cb3c1](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/2c6e3ed4-95f7-46e1-947c-1c1a9f5eeefd)
![282471309-5bf3641f-ac40-409b-b3e0-b0941e6cd3ba](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/095129b6-b08d-4580-b636-b95c5524ff1a)

![282471420-e0b27216-7013-42e9-b0c6-155c43c2c2b5](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/5ecbcd77-b7b7-4293-9e1b-584c964d83a7)


![282471495-6c6bbcb7-116f-4862-b7b4-c7b9960c38b7](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/6569f196-7453-4a31-a066-24db8ae3d57b)


![282471631-70d62334-d2dd-4c91-ba35-2b3c820260dd](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/798568ee-5dc9-4622-9680-6921e39898c2)
![282471729-3c4ec200-8cc0-45bc-894f-4a6e778bd7a4](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/bc92f4df-c4c0-4139-9a8e-f2d3f6d94a39)

![282471899-1395f06d-9c01-4a53-9b6f-893217f6631a](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/486042af-f84c-48ff-9542-c64b1534baa4)



![282472002-54a2638a-cb17-4ec8-b2b1-88b3aeb2c296](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/b5a491b8-c41e-45ed-834b-4b8c79b1f3e6)

![282472080-1cf9853e-d725-4320-9e6b-c31a4638a77c](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/cb2d4747-7e2f-4799-974c-4730660df875)

![282473422-4d525acf-457a-4535-88ba-4da10a93c42b](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/d06873bf-b073-45da-9353-46dd1a1bca8f)

![282473539-c313f8c4-53c8-45ca-ab6b-03c2b5bb76c7](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/67f94ed5-26fe-42fc-9735-9ae753559a93)

![282473686-53bc4c0f-d3f4-4de9-a5dd-a5da60ad7df0](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/0d126a0e-bb60-4fcb-b823-ab2d1ac86a60)


![282474045-6e378e75-7a45-47ab-b4a0-151b2ff40d8c](https://github.com/divyakumars/ODD2023-Datascience-Ex-07/assets/119393621/2571cbf9-a7cf-4d7d-b247-9833907d5ad5)


### RESULT :
Thus, the various feature selection techniques have been performed on a given dataset successfully.






