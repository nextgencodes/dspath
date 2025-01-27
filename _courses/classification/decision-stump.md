---
title: "Decision Stumps: The Simplest Yet Powerful Building Block in Machine Learning"
excerpt: "Decision Stump Algorithm"
# permalink: /courses/classification/decision-stump/
last_modified_at: 2022-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Machine Learning
tags: 
  - Machine Learning
  - Classification Model
  - Decision Stump
---

{% include download file="decision_stump.ipynb" alt="download decision tree code" text="Download Code" %}

## Introduction: What is a Decision Stump?

Imagine you're trying to decide whether to bring an umbrella with you. You might think, "Is it cloudy?" If the answer is "yes," you grab the umbrella; if "no," you leave it. A decision stump is like this simple decision-making process, but for computers.

A **Decision Stump** is the most basic form of a decision tree. It's a machine learning model that makes a decision based on the value of a single input feature. Think of it as a single "if-then-else" rule. Despite their simplicity, decision stumps are powerful building blocks and can be very useful in complex models when combined with other techniques.

**Real-World Examples:**

*   **Spam Email Detection:** A decision stump might ask, "Does the email contain the word 'viagra'?" If yes, it might be flagged as spam.
*   **Credit Risk Assessment:** A decision stump might ask, "Is the credit score below 600?" If yes, the applicant might be considered high risk.
*   **Medical Diagnosis:** A decision stump might ask, "Is the patient's temperature above 38 degrees Celsius?" If yes, the patient may have a fever.

These simple rules, when combined, can create more complex and powerful decision-making systems.

## The Math Behind Decision Stumps

Decision stumps, at their core, use simple mathematical comparisons to make decisions. The goal is to find the "best" way to split the data based on a single feature, minimizing error.

**Classification:**
For a classification problem (like spam detection), a decision stump determines a threshold, `t`, on a single feature and predicts the class based on whether the feature's value is above or below that threshold.

*   Let `x` be the input feature and `t` be the threshold.
*   The decision rule is:
    *   If `x < t`, predict class A.
    *   If `x >= t`, predict class B.

Mathematically, this looks like:

$$
\hat{y} =
\begin{cases}
A & \text{if } x < t \\
B & \text{if } x \geq t
\end{cases}
$$

*   $$ \hat{y} $$ is the prediction.
*   A and B are the possible classes that can be predicted.
*   `x` is feature in consideration.
*   `t` is the threshold value.

For Example:

If we want to predict if an email is spam or not based on the number of words in the email.
Let the feature be x = number of words in the email, and threshold value is t = 100.
If x<100 the email is not spam(A). If x>=100 then email is spam (B).

**Regression:**
For a regression problem (like predicting house prices), the decision stump predicts a constant value based on which side of the threshold, `t`, the feature falls.

*   Let `x` be the input feature, `t` the threshold, `c1` the prediction below the threshold and `c2` the prediction above the threshold.
*    The decision rule is:
    *   If `x < t`, predict `c1`.
    *   If `x >= t`, predict `c2`.

Mathematically:
$$
\hat{y} =
\begin{cases}
c1 & \text{if } x < t \\
c2 & \text{if } x \geq t
\end{cases}
$$

where,
* $$ \hat{y} $$ is the predicted output
* `x` is feature in consideration
* `t` is the threshold value.
* `c1` is predicted value when x<t
* `c2` is predicted value when x>=t

For Example:
If we want to predict the house price based on its area.
Let x = area of the house, t = 1000sqft
c1 = 200000(USD) if x<1000, if x>=1000 c2 = 400000(USD)

**Finding the Best Split:**

The algorithm tries different thresholds ( `t`) and chooses the one that best separates the data using a criteria like Gini impurity (for classification) or mean squared error (for regression).

## Prerequisites and Preprocessing

**Assumptions:**

*   Decision stumps don't make strong assumptions about data distribution. They're non-parametric models.
*   They are designed for data that can be split using a single feature with a threshold.

**Preprocessing:**

*   **No Strict Normalization Requirement**: Unlike models such as linear regression, decision stumps don't require data normalization or standardization because the comparison of feature values with the threshold is scale-invariant. However, you might want to scale the data if it helps your threshold selection process (i.e. make thresholds easier to select).
*   **Handling Missing Values**: Decision stumps can be adapted to handle missing values by creating additional branches for them. This is not always necessary because a single feature is evaluated and thresholded so missing values for other features should not affect this model. It is usually handled by imputation.
*   **Categorical Data:** Categorical data needs to be converted to numeric data (e.g., using one-hot encoding or ordinal encoding) before using decision stumps.

**Python Libraries:**

*   `scikit-learn`: The main library for machine learning in Python. Specifically, `sklearn.tree.DecisionTreeClassifier` and `sklearn.tree.DecisionTreeRegressor` can be used to create decision stumps with `max_depth=1`.
*   `pandas`: For handling data, reading data in to dataframes and data preprocessing.

## Tweakable Parameters and Hyperparameters

Decision stumps have few hyperparameters, as they are very simple models, yet, they can greatly affect the behavior of the model:

*   **`max_depth`:** The maximum depth of the decision tree. For a decision stump, `max_depth` should be set to `1`. A value greater than 1 means the model will no longer be a decision stump. This needs to be exactly 1 for a decision stump.
    *   **Effect:** If you increase this value, you wont get a decision stump anymore.

*   **`criterion` (for Classification):** The function used to measure the quality of a split. Common options are:
    *   `gini`: Uses the Gini impurity to measure impurity.
    *   `entropy`: Uses the information gain(entropy) to measure the purity.
    *   **Effect:** Different criterions can produce slightly different splits in the data and can slightly change model performance. Usually the change is negligible.

*  **`splitter`** (for Classification): Strategy used to choose the split at each node.
  * `best`: Choose the best split.
  * `random`: Choose a random split
  * **Effect:** `Best` is a deterministic strategy, and `random` can lead to more variation in model behaviour.

*   **`min_samples_split`:** The minimum number of samples required to split an internal node.
    *   **Effect:** Higher values lead to less complex models. For a decision stump, this may have an effect on the splitting of data if there are not enough samples.

*   **`min_samples_leaf`:** The minimum number of samples required to be at a leaf node.
    *   **Effect:** Higher values lead to less complex models and prevent overfitting.

*   **`max_features`:** The number of features to consider when looking for the best split.
     *  **Effect:** This is not directly applicable for DecisionStumps because they can split using only 1 feature, but it can be useful during model exploration and selection of best feature for the DecisionStump model.

*   **`random_state`:** Used for reproducibility.
    *   **Effect:** Sets seed for random operations, ensuring you get the same results.

*  **`max_leaf_nodes`** Grow tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
    *  **Effect:** This controls the number of leaf nodes, higher the value more complex the model, it will never grow bigger than a depth of 1 when used in decision stump.

## Data Preprocessing

As mentioned before, decision stumps don't always need preprocessing. Here are the key points:

*   **Scaling Not Required**: Because decision stumps are based on simple comparisons with the threshold value, they are not scale dependent.
*   **Categorical Data Encoding**: Encoding categorical variables is required so that model can threshold on a single numeric column. For example, if a dataset has a color variable which takes values such as red, green, blue. Then this needs to be encoded before model fitting, like one hot encoding into 3 different columns or converting them into ordinal values 1, 2, 3.
*  **Missing value Imputation**: Missing values need to be handled before fitting a model. A decision stump model can be created which handles missing value as a feature as well, but that model will not be a decision stump in strict sense.
* **Feature Selection:** The user can select which feature is the most useful for splitting the data, and can use that single feature in the decision stump model.

**Examples:**

*   **Data with varying scales:** If you have data with both values between 0-1 and values between 100-1000 then those values need not be scaled because the decision stump is able to find the optimal threshold even with varying scales in different features.
*   **Categorical features:** If you have a feature with categories such as `low`, `medium`, and `high` then this feature needs to be converted to numerical data using ordinal or one hot encoding before a decision stump can use this feature.

## Implementation Example

Let's implement a simple Decision Stump in Python using `scikit-learn` with some dummy data:

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
```


```python
# Dummy data
data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]}

df = pd.DataFrame(data)
df.head()
```
Output:

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
      <th>feature1</th>
      <th>feature2</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split data into features and target
X = df[['feature1']]
y = df['target']
#Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
```


```python
# Create Decision Stump model
model = DecisionTreeClassifier(max_depth=1, random_state=42)
```


```python
# Train the model
model.fit(X_train, y_train)
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
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(max_depth=1, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>DecisionTreeClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeClassifier.html">?<span>Documentation for DecisionTreeClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeClassifier(max_depth=1, random_state=42)</pre></div> </div></div></div></div>




```python
# Make predictions
y_pred = model.predict(X_test)
y_pred
```
Output:
```
    array([1, 0, 1])
```



```python
# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test,y_pred)
print(f"Accuracy: {accuracy}")
print("Classification report: \n", report)
```
Output:
```
    Accuracy: 0.6666666666666666
    Classification report: 
                   precision    recall  f1-score   support
    
               0       1.00      0.50      0.67         2
               1       0.50      1.00      0.67         1
    
        accuracy                           0.67         3
       macro avg       0.75      0.75      0.67         3
    weighted avg       0.83      0.67      0.67         3
``` 


```python
#Save Model
filename = 'decision_stump_model.pkl'
pickle.dump(model, open(filename, 'wb'))
```


```python
#Load Model
loaded_model = pickle.load(open(filename, 'rb'))
print("Loaded model prediction",loaded_model.predict(X_test))
```
Output:
```
    Loaded model prediction [1 0 1]
``` 

**Explanation:**

*   **Accuracy:** The accuracy score is approximately 0.67, or 67%. This means that out of the three data points in the test set, the model correctly classified two. Accuracy is calculated by (TP+TN)/(TP+TN+FP+FN)
    
*   **Classification Report:**
    
    *   **Precision:**
        
        *   For class 0 (the first row), precision is 1.00, meaning all predictions made for class 0 were correct. TP/(TP+FP). There were no false positives, meaning all predicted class 0s were actually class 0.
            
        *   For class 1 (the second row), precision is 0.50. Meaning only half of the predicted class 1 were actually class 1. TP/(TP+FP)
            
    *   **Recall:**
        
        *   For class 0, recall is 0.50, meaning the model correctly identified half of the actual class 0 instances. TP/(TP+FN)
            
        *   For class 1, recall is 1.00, meaning the model correctly identified all the actual class 1 instances. TP/(TP+FN)
            
    *   **F1-score:** The F1-score, which is the harmonic mean of precision and recall, is 0.67 for both class 0 and class 1. 2\*(Precision\*Recall)/(Precision+Recall)
        
    *   **Support:** The number of actual occurrences of each class in the test set. There are two instances of class 0 and one instance of class 1 in the test set.
        
    *   **accuracy:** The accuracy score of the model.
        
    *   **macro avg**: Macro average of precision, recall, and f1-score.
        
    *   **weighted avg**: Weighted average of precision, recall, and f1-score, weighted by the support of each class.
        
*   **Pickle:** pickle.dump is used to save the model to a file and pickle.load loads the saved file for later use.
    
*   **Loaded model prediction:** This shows the output of the loaded model.

Post-Processing
---------------

*   **Feature Importance:** For decision stumps, the feature selected by the model and the threshold can give useful information about that particular feature in the dataset. In the example given above the most important variable is feature1.
    
*   **AB testing:** Since the model is so simple AB testing is usually not applicable here, however if there are multiple features that can be used for creating decision stumps, then AB testing for comparing which feature gives the best result can be used.
    
*   **Hypothesis Testing:** Hypothesis testing could be used to validate if the split provided by the stump is statistically significant.
    
*   Other statistical tests like Chi-square can be used to validate the result of splits.
    

Hyperparameter Tuning
---------------------

Hyperparameter tuning for decision stumps is usually not necessary as they are very simple. However, if you want to find the best threshold, or to compare the best split, you can use different hyperparameters and evaluate different decision stump models. The model is sensitive to the feature selection and it is important to use the best feature for training the decision stump model.

```python
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Dummy data
data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]}

df = pd.DataFrame(data)

# Split data into features and target
X = df[['feature1','feature2']]
y = df['target']
#Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
# Define Hyperparameter grid
params = {'max_depth':[1],
          'criterion':['gini', 'entropy'],
         'splitter': ['best', 'random']
          }

# Model with grid search for different features
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid=params, cv=2)
grid.fit(X_train,y_train)

print(f"Best parameters for DecisionStump: {grid.best_params_}")
print("Best Score for Decision Stump: ", grid.best_score_)
```

The best score tells the quality of the model based on different hyperparameters. Here, the user can decide which hyperparameter values and feature is giving best results.

Checking Model Accuracy
-----------------------

Model accuracy is checked using several metrics:

*   **Accuracy**: As discussed previously is total correct predictions / total predictions. (TP+TN)/(TP+TN+FP+FN).
    
*   **Precision:** Measures the proportion of true positives among all positive predictions. TP/(TP+FP)
    
*   **Recall (Sensitivity):** Measures the proportion of true positives among all actual positives. TP/(TP+FN)
    
*   **F1-Score**: Harmonic mean of precision and recall 2\*(Precision\*Recall)/(Precision+Recall).
    
*   **AUC-ROC:** (Area Under the Curve - Receiver Operating Characteristic). This metric is useful for checking the overall model quality where we measure performance of the model by calculating area under the ROC curve.
    
*   **Confusion Matrix:** A table that summarizes the performance of the classification algorithm which shows number of correct and incorrect predictions by showing the values of TP, TN, FP and FN.
    

Productionizing Steps
---------------------

*   **Local Testing:** Create a test script to make sure model is loading correctly, predicting values correctly, output is of expected types, and there are no run time errors.
    
*   **On-Prem:** Create containers using docker and deploy the docker images on the server. Make sure the security requirements are met. Create endpoint or API for model invocation.
    
*   **Cloud:** Deploy the model on cloud platform and use cloud provided tools to test model. Deploy on a serverless platform or container platform as necessary. Ensure that proper authorization and authentication is in place.
    
*   **Real time and Batch:** Set up pipeline for ingestion of new data into the model for predictions and monitoring of the production system. Create a pipeline for batch prediction which runs at scheduled interval.
    

Conclusion
----------

Decision stumps might seem overly simplistic but they form the fundamental building blocks for complex machine learning methods. They are used in ensemble algorithms like Boosting and Random Forests, where multiple decision stumps are combined.

While they are not typically used as standalone models, they serve as a foundational model to learn complex machine learning concepts. Newer and more sophisticated models, such as deep learning models, have mostly replaced the use of decision stumps for more complex real-world problems. However, their simplicity and interpretability make them valuable for learning and understanding machine learning concepts.

Decision stumps are used in machine learning research, and are valuable for anyone looking to understand machine learning concepts.

##  References
    
1. Better trees: an empirical study on hyperparameter tuning of classification decision tree induction algorithms: [https://link.springer.com/article/10.1007/s10618-024-01002-5](https://link.springer.com/article/10.1007/s10618-024-01002-5)