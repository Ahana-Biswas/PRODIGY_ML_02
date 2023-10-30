# PRODIGY_ML_02
SECOND TASK OF MACHINE LEARNING.
Customer Segmentation using K-Means Clustering
Customer segmentation is the process of grouping a company's customers using shared characteristics so that the company can cater to each group effectively and appropriately. Machine learning models can process customer data and discover patterns difficult to spot through intuition and manual examination of data. K-means clustering is an unsupervised machine learning algorithm that uses an iterative process to divide an unlabeled dataset into k different clusters in such a way that each dataset belongs only to one group that has similar properties.

Summarizing the k-means algorithm
randomly choose k examples as initial centroids
while true:
  create k clusters by assigning each example to closest centroid
  create k new centroids by averaging examples in each cluster
  if centroids don't change:
    break
Notebook Summary
1. Importing necessary libraries
  set os.environ["OMP_NUM_THREADS"] = '1' to avoid multiple warnings in latter stages
2. Importing and viewing the dataset
  use pandas dataframe to view the dataset's quantity, available features, datatypes, and check for null values
3. Data Visualization
  (a) Distribution plots (histplots)
    checking densities of age, annual income, and spending
  (b) Categorical plots (violinplots)
    checking densities of age, annual income, and spending across genders
  (c) Relational plots (scatterplot)
    checking variation of spending with annual income
4. Finding Optimum no. of clusters
  use elbow method to find optimal no. of groups the model can create based on:
  (a) age and spending
  (b) annual income and spending
  (c) age, annual income, and spending
5. Plotting the clusters using optimal k
  use 2d and 3d plots to visualize clusters using optimal characteristics obtained in previous step
