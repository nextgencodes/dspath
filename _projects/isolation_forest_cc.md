---
title: "Fraud Detection in Credit Card Transactions with Isolation Forest"
last_modified_at: 2025-01-25T01:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories: [Data Science, Machine Learning, Anomaly Detection]
tags: [Python, Pandas, Scikit-learn, Isolation Forest, Data Visualization]
---
{% include download file="ccfraud_isolation_forest.ipynb" alt="download credit card fraud detection using isolation forest code" text="Download Code" %}

In this post, we'll explore how to use the Isolation Forest algorithm to detect anomalies in credit card transaction data. Anomaly detection is a crucial task in various domains, including fraud detection, network intrusion detection, and equipment failure prediction. Here, we focus on identifying unusual credit card transactions that might be indicative of fraud.

## Understanding the Data

We'll be using a credit card fraud dataset from Kaggle ([https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)). This dataset contains anonymized transaction data, including various features like `V1`, `V2`, etc., along with a `Class` label indicating whether a transaction is fraudulent (1) or not (0).

## The Python Code

Here is the Python code:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset and limit the number of rows to 40000 for efficiency
credit_data = pd.read_csv('data/creditcard.csv', nrows=40000)

# Display the first few rows of the DataFrame
print(credit_data.head())

# Scale all columns except 'Class' using StandardScaler
scaler = StandardScaler().fit_transform(credit_data.loc[:, credit_data.columns != 'Class'])

# Convert scaled data to a DataFrame
scaled_data = scaler[0:40000]
df = pd.DataFrame(data=scaled_data)

# Separate features and target variable
X = credit_data.drop(columns=['Class'])
y = credit_data['Class']


# Determine the fraction of outliers (fraudulent transactions)
outlier_fraction = len(credit_data[credit_data['Class'] == 1]) / float(len(credit_data[credit_data['Class'] == 0]))

# Create and fit the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=outlier_fraction, random_state=42)
model.fit(df)

# Predict outliers
scores_prediction = model.decision_function(df)
y_pred = model.predict(df)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

# Print the accuracy
print("Accuracy in finding anomaly:", accuracy_score(y, y_pred))

# Selecting the feature for the y-axis
y_feature = credit_data['Amount']

# Adding the predicted labels to the original dataset
credit_data['predicted_class'] = y_pred

# Plotting the graph
plt.figure(figsize=(7, 4))
sns.scatterplot(x=credit_data.index, y=y_feature, hue=credit_data['predicted_class'], palette={0: 'blue', 1: 'red'}, s=50)
plt.title('Visualization of Normal vs Anomalous Transactions')
plt.xlabel('Data points')
plt.ylabel(y_feature.name)
plt.legend(title='Predicted Class', loc='best')
plt.show()
```

Code Explanation
----------------

1.  **Import Libraries**: The code begins by importing essential libraries, such as pandas for data handling, numpy for numerical operations, seaborn and matplotlib for plotting, and scikit-learn for machine learning tasks.
    
2.  **Load Data**: The dataset is read using pd.read\_csv(), limiting the number of rows to 40000 for demonstration purposes. This line uses the relative path data/creditcard.csv, you need to make sure that you have the creditcard.csv file in that location relative to the location of your jekyll project.
    
3.  **Data Scaling**: The numerical features are standardized using StandardScaler to ensure each feature contributes equally to the distance calculations, which are critical for many machine learning algorithms. The class column is not scaled and only numerical features are used to scale data.
    
4.  **Define Features and Target:** The code separates the features (X) and the target variable (y). The target variable (y) here is the column Class, which tells us if a transaction was fraudulent.
    
5.  **Calculate Outlier Fraction:** The contamination parameter of the Isolation Forest is set to the fraction of fraudulent transactions within the dataset.
    
6.  **Train Isolation Forest:** An IsolationForest model is initialized with the contamination parameter. This model is then trained using the scaled data, the model learns to isolate the anomalies in the data.
    
7.  **Make Predictions:** The trained model is used to predict whether each transaction is an anomaly or not. The predictions are converted to 0s and 1s to match the original label column.
    
8.  **Evaluate Model**: The model's accuracy at detecting anomalies is then printed.
    
9.  **Visualize Results**: A scatter plot is created using seaborn, where points are colored based on whether they are classified as normal (blue) or anomalous (red) by the Isolation Forest. The plot shows how the model is classifying the data by using a feature called Amount on the y-axis and data points (row number) on the x-axis.
    

How Isolation Forest Works
--------------------------

The Isolation Forest algorithm is an unsupervised learning method that isolates anomalies. It works by randomly selecting a feature and then randomly selecting a split value between the minimum and maximum values of the selected feature. This is done recursively and the anomalous data points tend to get isolated much earlier in the process. The model assigns a score to each data point based on how many splits it took to isolate the data point. Anomalous data points tend to have lower scores.

Results
-------

Running the code will output:

*   The first five rows of the credit card transaction data
    
*   The accuracy of the anomaly detection model
    
*   A visualization of the predicted anomalies, which can be useful for understanding model performance and behavior.
    

You'll see that the accuracy is usually high as most of the data will not be an anomaly. The plot will highlight the predicted anomalous transactions.

Conclusion
----------

This blog post provides a practical demonstration of how to use the Isolation Forest algorithm for anomaly detection in credit card transaction data. This technique is valuable for real-world fraud detection systems. The Isolation Forest model is very effective at identifying anomalies and also provides a score that can be used for further analysis. Feel free to explore this code and dataset to gain a deeper understanding of anomaly detection methods.

