import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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
features=data_balanced.drop('target',axis=1)
targets=data_balanced['target']
features_train,features_vald_test,target_train,target_vald_test=train_test_split(features,targets,test_size=0.3)
features_vald,features_test,target_vald,target_test=train_test_split(features_vald_test,target_vald_test,test_size=0.5)
print(f'Training set size: {features_train.shape[0]}')
print(f'Validation set size: {features_vald.shape[0]}')
print(f'Testing set size: {features_test.shape[0]}')
scale=StandardScaler()
features_train = scale.fit_transform(features_train)
features_vald = scale.transform(features_vald)
features_test = scale.transform(features_test)
for k in range(1,21,3):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(features_train,target_train)
    target_predict=knn.predict(features_vald)
    score=accuracy_score(target_vald,target_predict)
    print(f'prediction:{target_predict}')
    print(f'actual value:{target_vald}')
    print(f'accuarcy for k={k} on validation set: {score:.4f}')



