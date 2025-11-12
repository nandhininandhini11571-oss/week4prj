import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("studentdata.csv")
print(df)
x=df[['python_marks','english_marks']]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
km=KMeans(n_clusters=2,random_state=77)
km.fit(x_scaled)
df['cluster']=km.labels_
print(df)
c1=df[df['cluster']==0]
print('cluster1')
print(c1)
c2=df[df['cluster']==1]
print('cluster2')
print(c2)