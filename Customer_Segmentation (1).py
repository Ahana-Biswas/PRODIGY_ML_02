#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = '1'


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("Mall_Customers.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(["CustomerID"], axis=1, inplace=True)
df.head()


# In[9]:


plt.figure(1, figsize=(15, 5))
n=0
for x in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
    n+=1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.1, wspace=0.5)
    sns.histplot(df[x], kde=True, stat="density", kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
plt.show()


# In[10]:


plt.figure(figsize=(15, 5))
sns.countplot(y="Gender", data=df)
plt.show()


# In[11]:


plt.figure(1, figsize=(15, 6))
n=0
for cols in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
    n+=1
    plt.subplot(1, 3, n)
    sns.set(style="whitegrid")
    plt.subplots_adjust(hspace=0.1, wspace=0.5)
    sns.violinplot(x=cols, y="Gender", data=df)
    plt.ylabel("Gender" if n==1 else "")
plt.show()


# In[12]:


sns.relplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df)


# In[14]:


X1 = df.loc[:, ["Age", "Spending Score (1-100)"]].values

from sklearn.cluster import KMeans

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker=8)
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[15]:


X2 = df.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10)
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker=8)
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[16]:


X3 = df.iloc[:, 1:]

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10)
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker=8)
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[17]:


kmeans = KMeans(n_clusters=4, n_init=10)
label = kmeans.fit_predict(X1)
plt.scatter(X1[:, 0], X1[:, 1], c=kmeans.labels_, cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black")
plt.title("Clusters of Customers")
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.show()


# In[18]:


kmeans = KMeans(n_clusters=5, n_init=10)
label = kmeans.fit_predict(X2)
plt.scatter(X2[:, 0], X2[:, 1], c=kmeans.labels_, cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black")
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()


# In[19]:


kmeans = KMeans(n_clusters=5, n_init=10)
clusters = kmeans.fit_predict(X3)
df["label"] = clusters

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c="blue", s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c="red", s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c="green", s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c="orange", s=60)
ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c="purple", s=60)
ax.view_init(30, 185)

plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")

plt.show()


# In[ ]:




