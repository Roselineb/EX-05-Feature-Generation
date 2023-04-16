# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
## Data.csv
```
import pandas as pd
df=pd.read_csv("data.csv")
df

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

oe=OrdinalEncoder()
df1=df.copy()

df1["City"] = oe.fit_transform(df1[["City"]])
df1["bin_1"] = oe.fit_transform(df1[["bin_1"]])
df1["Ord_1"] = oe.fit_transform(df1[["Ord_1"]])
df1["Ord_2"] = oe.fit_transform(df1[["Ord_2"]])
df1["bin_2"] = oe.fit_transform(df1[["bin_2"]])

df2=df.copy()

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
## Encoding.csv
```
import pandas as pd
qf=pd.read_csv("encoding.csv")
qf

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

oe=OrdinalEncoder()

qf1=qf.copy()


qf1["bin_1"] = oe.fit_transform(qf1[["bin_1"]])
qf1["nom_0"] = oe.fit_transform(qf1[["nom_0"]])
qf1["ord_2"] = oe.fit_transform(qf1[["ord_2"]])
qf1["bin_2"] = oe.fit_transform(qf1[["bin_2"]])

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
qf0=pd.DataFrame(sc.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf0   

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
qf2=pd.DataFrame(sc1.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
qf3=pd.DataFrame(sc2.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
qf4=pd.DataFrame(sc3.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf4
```
## Titanic_dataset.csv
```
import pandas as pd
rf=pd.read_csv("titanic.csv")
rf

#removing unwanted data
rf.drop("Name",axis=1,inplace=True)
rf.drop("Ticket",axis=1,inplace=True)
rf.drop("Cabin",axis=1,inplace=True)  

rf["Age"]=rf["Age"].fillna(rf["Age"].median())
rf["Embarked"]=rf["Embarked"].fillna(rf["Embarked"].mode()[0])

rf.isnull().sum()

rf1=rf.copy()

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
oe=OrdinalEncoder()

e1=OrdinalEncoder(categories=[embark])
rf1['Embarked'] = e1.fit_transform(rf[['Embarked']])
rf1['Sex'] = oe.fit_transform(rf[['Sex']])
rf1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
rf0=pd.DataFrame(sc.fit_transform(rf1),columns=['PassengerId', 'Survived', 'Pclass', 'Sex','Age','SibSp','Parch','Fare','Embarked'])
rf0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
rf3=pd.DataFrame(sc1.fit_transform(rf1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
rf3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
rf4=pd.DataFrame(sc2.fit_transform(rf1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
rf4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
rf5=pd.DataFrame(sc3.fit_transform(rf1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
rf5
```
# OUPUT
# Data.csv:
## Initial dataset:
# ![image](https://user-images.githubusercontent.com/128909895/232326659-8d89255d-3ac1-4650-9ccf-6d1cf02d38cd.png)
## Encoded dataset:
# ![image](https://user-images.githubusercontent.com/128909895/232326683-6edf7c2f-8cb0-4f78-9b38-02d2f1f6a362.png)

## Data scaling using MinMaxScaler:
# ![image](https://user-images.githubusercontent.com/128909895/232326704-1ff203ac-59e1-4377-a8d5-a452138b86be.png)
## Data scaling using MaxAbsScaler:
# ![image](https://user-images.githubusercontent.com/128909895/232326734-08c3be6f-8581-45ef-bfb0-7f42357eac6b.png)
## Data scaling using RobustScaler:
# ![image](https://user-images.githubusercontent.com/128909895/232326767-157f819c-7541-432c-976e-455899201f81.png)
# Encoding.csv:
## Initial dataset:
# ![image](https://user-images.githubusercontent.com/128909895/232326797-1b1678e1-03e0-445e-89d1-cd9127c36c08.png)
## Encoded dataset:
# ![image](https://user-images.githubusercontent.com/128909895/232326904-55f830a3-74d1-45eb-a8d9-6b081f2a4839.png)
## Data scaling using MinMaxScaler:
# ![image](https://user-images.githubusercontent.com/128909895/232326940-a7ebba0d-da91-4a04-89c3-9acc92a93475.png)
## Data scaling using StandardScalar:
# ![image](https://user-images.githubusercontent.com/128909895/232326979-b5ebc5ec-fca3-4cda-ab65-9f1e81d86b11.png)
## Data scaling using MaxAbsScaler:
# ![image](https://user-images.githubusercontent.com/128909895/232327487-4a85f079-9487-45d5-b305-91ce66a2918d.png)
## Data scaling using RobustScaler:
# ![image](https://user-images.githubusercontent.com/128909895/232327236-cf0faceb-f138-4bd4-9ed9-b11c0e85a015.png)


# Titanic_dataset.csv:
## Initial dataset:
# ![image](https://user-images.githubusercontent.com/128909895/232327069-47a1e14e-7465-4949-996e-f0c7fbac2bed.png)
## isnull.sum()
# ![image](https://user-images.githubusercontent.com/128909895/232327091-61b4ef6b-8d4b-4eaf-81bc-825587825240.png)
## Encoded dataset:
# ![image](https://user-images.githubusercontent.com/128909895/232327113-ef5f7ebe-b3e8-4f68-9517-8923be3e979d.png)
## Data scaling using MinMaxScaler:
# ![image](https://user-images.githubusercontent.com/128909895/232327133-d303816d-be3d-4401-93ed-5186034ea9ad.png)
## Data scaling using StandardScalar:
# ![image](https://user-images.githubusercontent.com/128909895/232327157-eedf176b-9851-4b28-9932-75cb99feca14.png)
## Data scaling using MaxAbsScaler:
# ![image](https://user-images.githubusercontent.com/128909895/232327188-49e02177-8050-4f16-8fe5-956cd09afad2.png)
## Data scaling using RobustScaler:
# ![image](https://user-images.githubusercontent.com/128909895/232327308-b27e2901-cd77-4f81-b6f0-23fe3bc4a429.png)
# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.



