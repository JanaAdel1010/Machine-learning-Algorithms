import pandas as pd
from sklearn.utils import resample
data=pd.read_csv("C:\School\Term 7\Introduction to Machine learning\Lab\Lab 1\ML- Assignment 1/magic04.data",delimiter=',',header=None)
data.columns=[f'feature{i}' for i in range(1,11)] + ['target']
data['target']=data['target'].map({'g':1,'h':0}).values
print(data.head())
class_h=data[data['target']==0]
class_g=data[data['target']==1]
numofh=class_h.shape[0]
class_g_balanced = resample(class_g, replace=False,n_samples=numofh,random_state=42) 
data_balanced=pd.concat([class_h,class_g_balanced])
print(data_balanced['target'].value_counts())
# data_balanced.to_csv("C:\School\Term 7\Introduction to Machine learning\Lab\Lab 1\ML- Assignment 1/magic04.csv",index=False)



