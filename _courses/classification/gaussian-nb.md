---
title: "Gaussian Naive Bayes: A Simple Classifier for Continuous Data"
excerpt: "Gaussian Naive Bayes Algorithm"
# permalink: /courses/classification/gaussian-nb/
last_modified_at: 2024-02-02T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Probabilistic Model
  - Supervised Learning
  - Classification Algorithm
tags: 
  - Probabilistic Models
  - Classification algorithm
  - Bayesian methods
---

{% include download file="gaussian_naive_bayes.ipynb" alt="download gaussian naive bayes code" text="Download Code" %}


## Introduction: What is Gaussian Naive Bayes?

Imagine you're trying to predict if a plant is a rose or a daisy based on its petal length and width. Gaussian Naive Bayes is a machine learning algorithm that can help with this. Instead of counting words like in text classification, this classifier focuses on continuous numerical features that can be assumed to be distributed according to a bell curve.

**Gaussian Naive Bayes** is a classification algorithm that uses probabilities to make predictions. It's a specific type of Naive Bayes that assumes that the continuous numerical features in your dataset follow a Gaussian (normal) distribution, also known as a bell curve. This assumption helps in calculating the probabilities used for classification, and this makes the algorithm relatively fast and easy to implement, though the assumption may not always hold true.

**Real-World Examples:**

*   **Medical Diagnosis:** Predicting the likelihood of a disease based on measurements like blood pressure, cholesterol levels, or body temperature.
*   **Customer Segmentation:** Categorizing customers based on spending habits, age, income, or other continuous variables.
*   **Weather Forecasting:** Predicting weather conditions like rain or no rain, based on continuous numerical features like temperature, humidity, and wind speed.
*   **Fraud Detection:** Identifying fraudulent transactions by looking at patterns in the transaction amount, location and date.

## The Math Behind Gaussian Naive Bayes

Gaussian Naive Bayes combines Bayes' theorem with the assumption that the data follows a Gaussian distribution.

**Bayes' Theorem:**

Bayes' theorem, as we discussed before, calculates the probability of a hypothesis, based on our prior knowledge:

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

Where:

*   `P(A|B)` is the probability of event A, given that B happened. This is also called the **posterior probability**.
*   `P(B|A)` is the probability of event B, given that A happened. This is called **likelihood**.
*   `P(A)` is the probability of event A happening. This is called the **prior probability**.
*   `P(B)` is the probability of event B happening. This is called the **evidence**.


{% include figure popup=true image_path="/assets/images/courses/Bayes_theorem_visual_proof.svg.png" caption="Bayes theorem" %}

**Gaussian Distribution:**

The Gaussian (normal) distribution describes how continuous data is distributed with a mean value and a standard deviation, and it is described by the following mathematical formula:

$$
P(x | \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

Where:

*   `P(x | μ, σ)` is the probability of feature x, with mean μ and variance σ.
*   `x` is the value of a feature.
*   `μ` (mu) is the mean (average) of the feature for a class.
*   `σ` (sigma) is the standard deviation of the feature for a class.
*   π (pi) is a mathematical constant approximately equal to 3.14159.
*   e is the base of the natural logarithm (approximately 2.71828)


{% include figure popup=true image_path="/assets/images/courses/Standard_deviation_diagram.svg.png" caption="Standard Deviation Diagram" %}

**How Gaussian Naive Bayes Works:**

1.  **Calculate Mean and Standard Deviation:** For each class, the algorithm calculates the mean (`μ`) and standard deviation (`σ`) for each continuous feature in the training data.
2.  **Calculate Likelihood:** When you want to classify new data, the algorithm uses the Gaussian formula to calculate the probability (likelihood) of each feature belonging to a particular class using the mean and standard deviation values.
3.  **Apply Bayes' Theorem:** The algorithm uses Bayes' theorem to calculate the posterior probability of each class by multiplying the likelihoods with the prior probabilities.
4.  **Make Prediction:** Finally, the algorithm selects the class with the highest probability.

**Example:**

Let's say you have to predict if a plant is a Rose or a Daisy, and the data you have is petal length (x) of different plants in centimeters:
1. Mean petal length of Roses (μ_rose) = 5cm and standard deviation(σ_rose)= 1cm
2. Mean petal length of Daisies (μ_daisy) = 3cm and standard deviation(σ_daisy)= 0.8cm
3. There are 3 Roses and 2 Daisies in our sample.
We need to find if a new plant with petal length x=4cm is a Rose or a Daisy.

*   **Prior Probability:**
    *   P(Rose) = 3/5 = 0.6
    *   P(Daisy) = 2/5 = 0.4

*   **Likelihood using Gaussian Distribution:**
    *   P(x=4 | Rose) = Use Gaussian formula with x=4, μ=5, σ=1 to get a probability value, lets say the value is 0.242
    *   P(x=4 | Daisy) = Use Gaussian formula with x=4, μ=3, σ=0.8 to get a probability value, lets say the value is 0.498

*   **Posterior Probability:**
    *   P(Rose | x=4) = P(x=4 | Rose) \* P(Rose) = 0.242 \* 0.6 = 0.1452
    *   P(Daisy | x=4) = P(x=4 | Daisy) \* P(Daisy) = 0.498 \* 0.4 = 0.1992

Since the probability of a plant being a Daisy is higher, the model predicts that the new plant is more likely to be a Daisy.

**Multivariate Gaussian Naive Bayes**

When there are multiple features to consider, it is assumed that the features are statistically independent and a class conditional density is defined as the product of the probability density of each individual feature. This can be represented as follows:

$$
P(C|x_1,x_2,...x_n) \propto P(C) \prod_{i=1}^{n} P(x_i|C)
$$

Where,
`P(C|x1,x2,...xn)` is the probability of a given class C, with features `x1, x2, ... xn`
`P(xi|C)` is the likelihood for feature `xi`, given the class `C`.
`P(C)` is the prior probability.

## Prerequisites and Preprocessing

**Assumptions:**

*   **Gaussian Distribution:** The key assumption is that numerical features follow a normal (Gaussian) distribution for each class. You can use histograms or Q-Q plots to test this assumption. However, the model still performs reasonably well even with this assumption not fully met.
*   **Feature Independence:** The features are assumed to be independent of each other given the class.

**Preprocessing:**

*   **Numerical Data:** Gaussian Naive Bayes is designed for continuous numerical features, which means that categorical data will not work and need to be converted to numerical data.
*  **Handling Skewness:** If data is heavily skewed, consider transformation using log transformation, box-cox transformation etc, so it follows closer to the normal distribution.
*   **Outlier Handling:** Outliers can affect the mean and standard deviation, so consider outlier detection and removal.
*   **Missing Value Imputation**: Missing values should be handled before training the model, it can be done using mean or median values. It can also be handled by removing rows that contains the missing data.
*   **Feature Scaling:** Although not strictly required, scaling or standardizing the features can help with numerical stability.

**Python Libraries:**

*   `scikit-learn`:  Provides the `GaussianNB` class for implementation.
*   `pandas`: For data manipulation.
*   `numpy`: For numerical calculations.
*  `matplotlib` and `seaborn`: For plotting graphs for checking data distribution.

## Tweakable Parameters and Hyperparameters

Gaussian Naive Bayes has a very small set of tweakable parameters:

*   **`var_smoothing`:** This parameter adds a portion of the largest variance of all features to the variance for calculation stability. It prevents division by zero in variance calculations, particularly when there are small sample sizes.
    *   **Effect:** Higher `var_smoothing` can make the model generalize better but with reduced variance, it reduces model complexity, and can reduce model performance on training set, but helps generalization on test set.
*  **`priors`**: Class priors. By default the model will use the prior from the train data, however user can also provide prior probabilities of each classes.
    *  **Effect:** If not set then prior will be set from the train dataset.
*  **`fit_prior`:** Boolean to specify if prior probability needs to be learned. By default it is set to True which will learn prior probability of each classes. If set to False a uniform prior will be used.
   *   **Effect:** If false, then uniform prior probability is used, which will give same weight to all the classes.

## Data Preprocessing

As discussed before data preprocessing is important in this model and here are some key points:

*   **Numerical Data:** This model needs numerical data, so categorical data needs to be converted to numerical.
*   **Data distribution:** The data distribution should be close to normal. Transformations like log transformation, square root, boxcox transformations can be used for making data normal.
*   **Missing values:** Missing values need to be imputed before model is trained so that all the features can be used by the model. Imputation can be performed by using mean or median values of that feature.
*  **Outliers**: Outliers should be removed or handled by other methods. Outliers can impact the Gaussian distribution and affect model accuracy.
*   **Feature Scaling:** Feature scaling such as normalization or standardization is not strictly required but can improve numerical stability, especially if the data is on very different scales.

**Examples:**

*   **Numerical Data:** If a model needs to be trained based on features like temperature, humidity, and wind speed then those features should be numerical data.
*  **Skewed data**: If data for any feature is highly skewed, then that data can be transformed using log, boxcox, yeo-johnson transformations. For example house prices are usually skewed, and transformation should be applied to make the distribution close to normal.
* **Outliers:** If outliers are present, then that needs to be removed or handled by using other techniques as it can affect model mean and standard deviation.
* **Scaling:** Although not strictly required, scaling the numerical features can make training faster and easier.

## Implementation Example

Let's implement a simple Gaussian Naive Bayes classifier using `scikit-learn` with some dummy data:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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


`Output:`

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
X = df[['feature1','feature2']]
y = df['target']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)
```


```python
# Create a Gaussian Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)
```


`Output:`

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
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GaussianNB</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.naive_bayes.GaussianNB.html">?<span>Documentation for GaussianNB</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GaussianNB()</pre></div> </div></div></div></div>




```python
# Make predictions
y_pred = model.predict(X_test)
print("model prediction: ", y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification report: \n", report)
```
`Output:`
```

    model prediction:  [0 0 1]
    Accuracy: 1.0
    Classification report: 
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         2
               1       1.00      1.00      1.00         1
    
        accuracy                           1.00         3
       macro avg       1.00      1.00      1.00         3
    weighted avg       1.00      1.00      1.00         3
    
 ```   


```python
# Save the model
filename = 'gaussian_naive_bayes_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Load the model
loaded_model = pickle.load(open(filename, 'rb'))
print("Loaded model prediction:", loaded_model.predict(X_test))
```
`Output:`
```
    Loaded model prediction: [0 0 1]
   ``` 

**Explanation:**

*   **Accuracy:** The accuracy score is 1.0, or 100%. Meaning all data points are predicted correctly. (TP+TN)/(TP+TN+FP+FN)
    
*   **Classification Report:**
    
    *   **Precision:** For class 0 the precision is 1.0, which means that out of all the points that are predicted as 0 all were actually 0. For class 1 the precision is 1.0, which means that all the points predicted as 1 were actually 1. TP/(TP+FP)
        
    *   **Recall:** For class 0 the recall is 1.0, meaning that the model was able to find all actual 0. For class 1 the recall is 1.0, which means all the actual 1's were identified by the model. TP/(TP+FN)
        
    *   **F1-Score:** F1-score is 1.0 for both the classes, which is harmonic mean of precision and recall. 2\*(Precision\*Recall)/(Precision+Recall)
        
    *   **Support:** The number of instances of each class in the test set.
        
    *   **accuracy:** The accuracy score of the model.
        
    *   **macro avg**: Macro average of precision, recall and f1-score.
        
    *   **weighted avg**: Weighted average of precision, recall and f1-score, weighted by the support of each class.
        
*   **Pickle:** Used to save and load the model using pickle.dump and pickle.load respectively.
    

Post-Processing
---------------

*   **Feature Importance:** Not directly available in GaussianNB, but the mean and standard deviation values for each feature per class can provide insight into which features are more important for classification.
    
*   **AB Testing:** Different data preprocessing steps can be compared using AB tests for better performance.
    
*   **Hypothesis Testing:** Evaluate the model performance with different statistical tests to ensure that the result is significant.
    
*   Other statistical tests can also be used for post-processing.
    

Hyperparameter Tuning
---------------------

```python
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)

# Define a grid of hyperparameters
params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

# Grid search for hyperparameter tuning
grid = GridSearchCV(GaussianNB(), params, cv=2)
grid.fit(X_train, y_train)

print(f"Best parameters for Naive Bayes: {grid.best_params_}")
print("Best Score for Naive Bayes: ", grid.best_score_)
```

`Output:`
```
Best parameters for Naive Bayes: {'var_smoothing': 1e-09}
Best Score for Naive Bayes:  0.5833333333333333
```

Checking Model Accuracy
-----------------------

Model accuracy can be checked using these metrics:

*   **Accuracy:** Total correct predictions / total predictions (TP+TN)/(TP+TN+FP+FN)
    
*   **Precision:** Fraction of true positives among all positives TP/(TP+FP)
    
*   **Recall:** Fraction of true positives among all actual positives. TP/(TP+FN)
    
*   **F1-Score:** Harmonic mean of precision and recall. 2\*(Precision\*Recall)/(Precision+Recall)
    
*   **AUC-ROC:** Area Under the Receiver Operating Characteristic curve.
    
*   **Confusion Matrix**: Table showing the performance of the model with values of True Positive, False Positive, True Negative, and False Negative.
    

Productionizing Steps
---------------------

*   **Local Testing:** Create test script to check that the model is performing correctly.
    
*   **On-Prem:** Use docker to create a container with the model and deploy on your local server.
    
*   **Cloud:** Use cloud provider tools to train, deploy, and monitor the model.
    
*   **Real time and Batch:** Set up a pipeline for ingestion of data for real time or batch processing.
    
*   **Monitoring:** Monitor model performance in production.
    

Conclusion
----------

Gaussian Naive Bayes is a simple and effective classifier, particularly when dealing with continuous numerical data that has (or can be transformed into) a normal distribution. While it is fast, its performance is largely dependent on the assumption of feature independence and the Gaussian distribution of features. It is widely used as a baseline model for many classification problems. Despite its simplicity, it can be a useful model when data distribution is normal, or can be transformed to be close to normal. There are newer and optimized algorithms, which perform better than Naive Bayes, such as deep learning algorithms.

References
----------

1.  **Scikit-learn Documentation:** [https://scikit-learn.org/stable/modules/naive\_bayes.html](https://www.google.com/url?sa=E&q=https://scikit-learn.org/stable/modules/naive_bayes.html)
    
2.  **Wikipedia:** [https://en.wikipedia.org/wiki/Naive\_Bayes\_classifier](https://www.google.com/url?sa=E&q=https://en.wikipedia.org/wiki/Naive_Bayes_classifier)