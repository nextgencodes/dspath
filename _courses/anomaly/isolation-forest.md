---
title: "Isolation Forest: Finding Needles in a Haystack"
excerpt: "Isolation Forest Algorithm"
last_modified_at: 2025-01-25T01:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Machine Learning
  - Anomaly Detection
tags: 
  - Machine Learning
  - Anomaly Detection
---


{% include download file="isolation_forest.ipynb" alt="download isolation forest code" text="Download Code" %}

Have you ever wondered how banks detect fraudulent transactions, or how manufacturers identify faulty products on an assembly line? These are real-world problems where we need to find the "odd ones out" – the anomalies. One powerful tool for this is the **Isolation Forest** algorithm. Let's dive in and see what it's all about!

## What is Anomaly Detection?

Imagine you're looking at a field of green grass. Suddenly, you spot a bright red flower. That red flower stands out because it's different from the rest. In data science, "anomalies" or "outliers" are data points that are different from the majority of your data.

**Real-World Examples:**

*   **Credit Card Fraud:** When someone uses your credit card in a location you've never been to, it's flagged as an anomaly.
*   **Manufacturing Defects:** A product that doesn't meet quality standards during manufacturing is an anomaly.
*   **Network Security:** A sudden surge in network traffic that is unusual is an anomaly that may indicate a cyberattack.
*   **Medical Diagnosis:** An unusual measurement in a patient's blood test could indicate an illness.

## How Does Isolation Forest Work?

The Isolation Forest algorithm takes a very unique approach to finding these oddities. Instead of trying to profile 'normal' data, it focuses on directly isolating 'anomalous' data points. Here's the basic idea:

1.  **Random Partitioning:** The algorithm randomly selects a feature (a column in your data) and then randomly picks a split value within the range of that feature. Think of it like cutting a cake into slices using random cuts.
2.  **Isolation:** This process is repeated to 'separate' or 'isolate' data points.
3.  **Anomalies are Easier to Isolate:** Data points that are anomalous will generally get isolated much faster – they are like small, easy-to-pick-out pieces of cake compared to larger, more complex chunks. Normal data points require more splits to be isolated.
4.  **Path Length:** The algorithm measures the number of splits it took to isolate a point. This is called the "path length". Shorter paths indicate anomalies.

## The Mathematics Behind It (Simplified)

Let's simplify the math without getting lost in complicated equations. The main concept involves calculating the **average path length** and the **anomaly score**.

* **Path Length (h(x)):** Imagine each data point as a point in a data space (think of a dot in a room). The number of splits required to isolate the point (or put a box around the point) is its path length. Anomalies will have shorter path lengths.

* **Average Path Length (c(n)):** If we had to isolate n points, this is average path length. We don't need to worry about the exact value but you can see it as the average path length required to isolate points.

* **Anomaly Score (s(x, n)):** This score is calculated from the path length. Think of it as a score for how abnormal the point is.

The anomaly score is given by the formula:

$$
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
$$


Where:

*   ___s(x,n)___ is the anomaly score of data point x in a dataset of size n.
    
*   ___E(h(x))___ is the average path length for data point x.
    
*   ___c(n)___ is the average path length needed to isolate a data point in a dataset of size n
    

Let's break this down:

1.  **2 raised to something**: Imagine you are playing a game where you get 2 points for each level up. So if you go up 2 levels you have 2\*2 = 4 points or 2 raised to power of 2.
    
2.  **Negative Power**: If we get 2 points for going up, we can say we go down by dividing the points. In this case it is just dividing by 2. So -1 means we divide by 2. -2 means we divide by 2, and again by 2, or 4.
    
3.  **E(h(x))/c(n)**: We already know what E(h(x)) and c(n) is. We can consider it as some number divided by other number.
    
4.  **Putting it Together**: If the path length E(h(x)) is very small, then the whole fraction is small. If we have a negative of that number then the number increases, so the value $$ 2^{-\\frac{E(h(x))}{c(n)}} $$ increase to close to 1. Which means, it is highly likely an anomaly.
    

**In simple terms:** The formula calculates a score between 0 and 1. If the score is closer to 1, the data point is likely an anomaly. If it's close to 0, the data point is considered normal.

Prerequisites for Using Isolation Forest
----------------------------------------

Before implementing the algorithm, here are a few things to consider:

1.  **No Strong Assumptions:** Isolation Forest makes fewer assumptions about the data distribution than many other anomaly detection methods. You don't need to assume your data is normally distributed.
    
2.  **Numerical Data:** The algorithm works best with numerical data. If you have categorical data, you'll need to encode it into numerical form.
    
3.  **No Prior Labeling:** Isolation Forest is an unsupervised learning algorithm. This means you don't need to have labeled "normal" and "anomaly" data to train your model.
    
4.  You can install these with pip using:
    
    *   ***scikit-learn*** for the Isolation Forest implementation.
        
    *   ***numpy*** for numerical operations.
        
    *   ***pandas*** for data manipulation.
        
    *   ***matplotlib and seaborn*** for visualization.

    ```bash
    pip install scikit-learn numpy pandas matplotlib seaborn
    ```

Tweakable Parameters and Hyperparameters
----------------------------------------

Here are some important parameters you can adjust in your Isolation Forest model:

*   **n\_estimators**: The number of isolation trees to build.
    
    *   **Effect:** More trees can lead to more robust results, but may also increase computational time.
        
    *   **Example:** If you have very complex data with lots of outliers, increase n\_estimators.
        
*   **max\_samples**: The number of data points to use to build each tree.
    
    *   **Effect:** Increasing this value can make the model better at detecting more global outliers, but less sensitive to local outliers.
        
    *   **Example:** For a dataset with many global and fewer local outliers, try setting max\_samples close to the size of the dataset.
        
*   **contamination**: The proportion of outliers you expect in your data.
    
    *   **Effect**: Affects the threshold for deciding whether a point is an anomaly or not.
        
    *   **Example:** If you know your data has about 5% outliers, set contamination=0.05.
        
*   **max\_features**: The maximum number of features to consider when making a split in each tree.
    
    *   **Effect:** If your dataset has too many features, then you can limit the features. If set to 1, then the tree will only use single feature.
        
    *   **Example:** If you want the algorithm to use some of the features but not all of them you can limit it.
        
*   **random\_state**: For reproducible results.
    
    *   **Effect:** Setting a random state ensures that your results are consistent every time you run your code.
        
    *   **Example:** Set random\_state=42 for consistent results.
        

Data Preprocessing
------------------

Isolation Forest generally does not require extensive data preprocessing like data normalization, but here are some things to consider:

*   **Feature Scaling:** Feature scaling (like normalization or standardization) is usually not necessary because Isolation Forest uses random splits. However, if different features have very different scales, consider scaling if it increases the model performance.
    
*   **Categorical Data Encoding:** As mentioned before, you need to convert categorical features into numerical values using techniques like One-Hot Encoding or Label Encoding.
    
*   **Handling Missing Values:** Missing values can disrupt the algorithm. You might need to impute missing values using techniques like mean imputation or median imputation.
    
*   **Feature Selection:** Selecting relevant features can improve the model and reduce training time.
    

**Example of Preprocessing:** Suppose you have a dataset with student data, including age and grade\_level. Age is a numeric feature that may vary from 5 to 25. Grade\_level may have the value K, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.

In this case you can one hot encode the Grade\_level feature into 13 individual features, for the values K, 1, 2, ..., 12. You can ignore scaling for age column. But it is not required.

Implementation Example
----------------------

Let's implement Isolation Forest using some dummy data.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Generate some dummy data with anomalies
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
X = np.concatenate([X, X_outliers], axis=0)
```


```python
# Create a dataframe
df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df.head()
```


output


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
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.149014</td>
      <td>-0.041479</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.194307</td>
      <td>0.456909</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.070246</td>
      <td>-0.070241</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.473764</td>
      <td>0.230230</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.140842</td>
      <td>0.162768</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fit Isolation Forest model
model = IsolationForest(n_estimators=100, random_state=42, contamination=0.17)
model.fit(df)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>IsolationForest(contamination=0.17, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>IsolationForest</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.IsolationForest.html">?<span>Documentation for IsolationForest</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>IsolationForest(contamination=0.17, random_state=42)</pre></div> </div></div></div></div>




```python
# Get the anomaly scores
scores = model.decision_function(df)
```


```python
# Get the anomaly prediction (1 for normal, -1 for anomaly)
predictions = model.predict(df)
```


```python
# Add predictions to the DataFrame
df['anomaly_score'] = scores
df['is_anomaly'] = predictions
```


```python
# Visualize results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='feature_1', y='feature_2', hue='is_anomaly', data=df, palette={1:'blue', -1:'red'})
plt.title('Isolation Forest Anomaly Detection')
plt.show()
```


    
{% include figure popup=true image_path="/assets/images/courses/if_output_7_0.png" caption="Anomaly Plot" %}

    



```python
# Print anomaly counts
print(df.is_anomaly.value_counts())
```
output
```

    is_anomaly
     1    99
    -1    21
    Name: count, dtype: int64
``` 


**Output and Explanation:**

1.  **Dummy data creation:** We created a dummy dataset using np.random.RandomState(42) which will generate random numbers always the same if initialized with same random state. This allows for reproducibility. We generated 100 data points that are normal and 20 outliers.
    
2.  **Model Fitting:** We create a IsolationForest model and fit the model using the dummy data. We used n\_estimators=100 and contamination = 0.17. This is because we know that there are about 20/120=0.17 outliers.
    
3.  **Anomaly Scores:** The decision\_function returns the anomaly scores. The output will be between 0 to 1, and lower scores indicate anomalies.
    
4.  **Anomaly Predictions:** The predict function outputs 1 for normal and -1 for anomaly.
    
5.  **Visualization:** The scatter plot shows that the outliers (red) are separated from the normal data points (blue).
    
6.  **Print Anomaly Counts:** The output will print how many anomalies (-1) and normal data points (1) are detected. For example:

Post-Processing and Model Evaluation
------------------------------------

After training your model, it's helpful to analyze results:

*   **Feature Importance:** Isolation Forest doesn't directly give feature importance, but you can analyze which features are frequently used in the splits and potentially affect outlier scores. For example, if the same feature is frequently used then that feature is a good predictor.
    
*   **Anomaly Threshold:** The decision boundary is automatically set by contamination. You might want to experiment with different contamination settings.
    
*   **A/B Testing:** If your system has an intervention, you can compare its effectiveness on populations with detected anomalies versus populations without detected anomalies.
    
*   Since Isolation Forest is unsupervised, traditional metrics like accuracy may not be appropriate. Here are some alternative metrics:
    
    *   **Precision, Recall, F1-Score (with ground truth):** If you have some known anomalies, you can calculate these scores by treating the problem as a binary classification problem. If you do not have ground truth, you can ignore it.
        
    *   **Precision** = TP/(TP+FP). (True positive / true positive + false positive). Out of all the points you marked as anomaly, how many are actually an anomaly.
        
    *   **Recall** = TP/(TP+FN). (True positive/ true positive + false negative). Out of all the actual anomalies, how many you identified.
        
    *   **F1-Score** = 2\*(Precision\*Recall)/(Precision+Recall). F1 score is the harmonic mean of precision and recall.
        
    *   **Area Under the ROC Curve (AUC-ROC)**: If you have some known anomalies, you can plot ROC curve and calculate the AUC.
        
    *   **Qualitative Assessment:** Examine the specific instances that the algorithm flags as anomalies, this can be useful in determining if the model is performing well in specific scenario.
        

Hyperparameter Tuning
---------------------

Hyperparameter tuning involves systematically finding the best parameters. Here's how to do it using GridSearchCV from scikit-learn:


```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_samples': ['auto', 0.5, 0.8],
    'contamination': [0.10, 0.15, 0.20]
}

# Create Isolation Forest model
iso_forest = IsolationForest(random_state=42)

# Set up grid search
grid_search = GridSearchCV(estimator=iso_forest, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')

# Fit the grid search
grid_search.fit(df[['feature_1', 'feature_2']])

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)
```
Output
```
Best parameters found:  {'contamination': 0.1, 'max_samples': 'auto', 'n_estimators': 50}
``` 


```python
grid_search.predict(df)
```


Output
```
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,
           -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,
            1])
```



```python
plt.plot(df)
plt.plot(grid_search.predict(df))
plt.legend(['feature1','feature2','prediction'])
```

{% include figure popup=true image_path="/assets/images/courses/if_output_12_1.png" caption="Features and prediction plot" %}
    
Model Productionizing
---------------------

Here are the steps to deploy the trained Isolation Forest model:

*   **Local Testing:** You can use the model on your local machine using python.
    
*   **Cloud Deployment:**
    
    *   **AWS, GCP, Azure:** You can deploy the model as an API endpoint using a cloud platform's machine learning services.
        
    *   **Docker:** Containerize the model for easier deployment.
        
*   **On-Premise:**
    
    *   **Dedicated Server:** Deploy the model on a dedicated server, with an API or scheduler.
        

Conclusion
----------

Isolation Forest is a powerful, versatile, and efficient algorithm for anomaly detection. Its ability to isolate anomalies directly makes it a good choice, and is being used in various industries like fraud detection, network security, and manufacturing.

While Isolation Forest has limitations, its ease of use, performance, and interpretability have cemented its place as a valuable tool. There are newer anomaly detection algorithms such as Autoencoders, and GAN-based anomaly detection, that provide better performance, depending on the use case.

Hopefully, this guide has demystified Isolation Forest and shown you how useful it can be in your projects.
