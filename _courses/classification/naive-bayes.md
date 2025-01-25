---
title: "Naive Bayes: A Simple and Effective Classifier for Text and More"
excerpt: "Naive Bayes Algorithm"
# permalink: /courses/classification/naive-bayes/
last_modified_at: 2024-01-10T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Machine Learning
  - NLP
tags: 
  - Naive Bayes
  - Classification
  - Text Analysis 
  - Machine Learning
  - Python
---

{% include download file="naive_bayes.ipynb" alt="download naive bayes code" text="Download Code" %}

## Introduction: What is Naive Bayes?

Imagine you're trying to guess if an email is spam. You might look for certain words like "lottery" or "free." Naive Bayes is a machine learning algorithm that works a bit like that, but much more systematically.

**Naive Bayes** is a classification algorithm that makes predictions based on probabilities. The term "Naive" refers to the assumption that all features in a dataset are independent of each other. Though this assumption is often not true in real life, the Naive Bayes algorithm often works remarkably well. It is particularly popular in Natural Language Processing (NLP), especially for text classification. Despite its simplicity, it can be a powerful tool for solving complex problems.

**Real-World Examples:**

*   **Spam Email Filtering:** As mentioned, it's widely used to identify spam emails based on words used in the email.
*   **Sentiment Analysis:** It can classify text as positive, negative, or neutral based on the words used.
*   **Document Categorization:** It can categorize documents into different categories (e.g., sports, politics, entertainment).
*   **Medical Diagnosis:** It can assist in medical diagnosis by using patient symptoms as features, and calculating the probability of them having a particular disease.

## The Math Behind Naive Bayes

Naive Bayes relies on probability and Bayes' theorem. Here's a simplified explanation:

**Bayes' Theorem:**

Bayes' theorem calculates the probability of an event (a hypothesis) based on prior knowledge of conditions that might be related to the event.

Mathematically:

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

Where:
*   `P(A|B)` is the probability of event A happening, given that event B has happened. This is the **posterior probability** that the Naive Bayes calculates for every class.
*   `P(B|A)` is the probability of event B happening, given that event A has happened. This is called the **likelihood**.
*   `P(A)` is the probability of event A happening. This is called the **prior probability**.
*   `P(B)` is the probability of event B happening. This is called the **evidence**.


{% include figure popup=true image_path="https://upload.wikimedia.org/wikipedia/commons/6/61/Bayes_theorem_tree_diagrams.svg" alt="Bayes Theorem" %}
`Bayes Theorem Tree Diagram`

**Naive Bayes Assumption:**

Naive Bayes assumes that all features are independent, which greatly simplifies the calculations.

Let's say we want to determine if an email is spam or not. The email content would be the feature.

*   Let A be the event that the email is spam,
*   Let B be the event that the email contains specific word such as "lottery"
Then `P(A|B)` is the probability that the email is spam, given that the email contains the word "lottery".
*   `P(B|A)` is the likelihood which is the probability that the email contains the word "lottery" given the email is spam.
*   `P(A)` is the probability of email being spam in general. This value depends on the proportion of spam emails in the data and is known as prior probability.
*   `P(B)` is the probability of any email having the word "lottery".

We calculate `P(A|B)` for each class and select the class which has maximum probability.

**Example:**

Suppose we're classifying emails as "spam" or "not spam."
Let's take the following simple data, and assume that we have following information about the data:
1. Number of emails that are spam (A) = 2
2. Number of emails that are not spam = 3
3. Number of spam emails with the word "free" (B) = 1.
4. Total number of emails = 5
5. Number of normal emails with the word "free" = 1.
6. Number of emails with "free" word = 2

We want to find P(spam| "free") = P(B|A)*P(A)/P(B)
* Prior Probability P(spam)= Number of spam emails/ Total Emails = 2/5
* Likelihood P("free"|spam) = Number of spam emails with free / Total number of spam emails = 1/2
* Evidence P("free") = Number of emails with free / Total number of emails = 2/5
P(spam|"free") = (1/2)\*(2/5)/(2/5) = 1/2 = 0.5
Therefore the probability of an email with the word "free" being spam is 0.5.

**Multivariate Naive Bayes**

If we have multiple features (for example the word "free", and "lottery" in the email). Then the probability can be calculated using product as:

$$
P(C | x_1, x_2, ..., x_n) = \frac{P(x_1 | C) * P(x_2 | C) *...*P(x_n | C) * P(C)}{P(x_1, x_2, ..., x_n)}
$$

Where,
*   `P(C|x1,x2, ... ,xn)` is the probability of a given class C based on features `x1, x2,...xn`
*   `P(xi|C)` is the likelihood for feature `xi`, for a given class `C`.
*   `P(C)` is the prior probability.

Since denominator `P(x1, x2, ..., xn)` is a constant, it is ignored and for simplicity the formula is written as:

$$
P(C | x_1, x_2, ..., x_n) \propto P(C) \prod_{i=1}^{n} P(x_i | C)
$$

In practice, we would calculate this for every possible class and select the class with maximum value.

## Prerequisites and Preprocessing

**Assumptions:**

*   **Feature Independence:** The core assumption is that all input features are independent of each other, which is often not the case in reality.
*   **Categorical Features:** Naive Bayes works well with categorical data, although the data can be converted to numerical data using different encoding schemes.
*   **Numerical Features:** For numerical features, Naive Bayes assumes that features follow a specific distribution, like Gaussian (normal), Multinomial, Bernoulli or others.

**Preprocessing:**

*   **Categorical Data Encoding**: Categorical features must be converted to numerical. Techniques like one-hot encoding, ordinal encoding, or feature hashing are common.
*   **Text Data Vectorization:** For text data, techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or CountVectorizer are essential. These transform the text into numeric representations.
*   **Handling Missing Values:** Missing values are generally handled by imputation with a suitable value (e.g., mean or median) or removing samples with missing data. This step is not strictly required for the model to work, but can help improve the model performance.
*   **Feature Scaling:** Feature scaling is usually not required as Naive Bayes calculates probabilities with the actual feature values.

**Python Libraries:**

*   `scikit-learn`: Includes various Naive Bayes implementations such as `GaussianNB`, `MultinomialNB`, and `BernoulliNB`.
*   `pandas`: Used for data manipulation, loading data, and handling dataframes.
*   `nltk`: For text processing, you will need to `pip install nltk`, then download the necessary datasets using the code:


Tweakable Parameters and Hyperparameters
----------------------------------------

Naive Bayes algorithms have a small set of hyperparameters:

*   **var\_smoothing** (GaussianNB): Adds a value to variance for smoothing, preventing zero variance.
    
    *   **Effect:** Higher value reduces model complexity and generalization, prevents overfitting.
        
*   **alpha** (MultinomialNB, BernoulliNB): Smoothing parameter, also known as Laplace smoothing or Lidstone smoothing. This is an additive (Laplace/Lidstone) smoothing parameter that prevents the zero probability issue.
    
    *   **Effect:** Higher value reduces model complexity and generalization, prevents overfitting. This helps prevent zero probabilities if a feature does not exist in a particular class in training data, but exist in the testing data.
        
*   **binarize** (BernoulliNB): Threshold for binarizing features.
    
    *   **Effect:** If None, the input is presumed to already consist of binary vectors. If a number then it binarizes the feature on the given threshold.
        
*   **fit\_prior** (all classes): Whether to learn class prior probabilities.
    
*   **Effect:** If False, a uniform prior will be used.
    

Data Preprocessing
------------------

Data preprocessing is very important for Naive Bayes, here are some key points:

*   **Scaling** is not required for the model as the model is based on probabilities and is not distance based.
    
*   **Encoding of Categorical Data** is important, and one hot encoding, ordinal encoding, feature hashing should be used for this purpose.
    
*   **Text vectorization** is required to convert the text into numerical features that Naive Bayes algorithm can understand.
    
*   **Missing Value Imputation** is required so that probabilities can be calculated correctly.
    

**Examples:**

*   **Text data:** If we want to classify emails into spam or not spam then that text data needs to be converted into numerical features.
    
*   **Categorical data:** If we want to predict if the user will purchase a product based on features such as "color," then the color feature with values such as red, blue, green will need to be converted into numerical features using techniques such as one hot encoding, ordinal encoding etc.
    
*   **Missing values:** If any of the features is missing a particular value then that needs to be imputed with some reasonable value so that the model is able to perform calculations. For example we can impute the missing values with mean of the feature.
    

Implementation Example
----------------------

Let's implement a simple Multinomial Naive Bayes classifier for text data using scikit-learn with dummy data:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
```


```python
# Download necessary nltk resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')
```
`Output:`
```
    [nltk_data] Downloading package punkt_tab to
    [nltk_data]    ...\nltk_data...
    [nltk_data]   Unzipping tokenizers\punkt_tab.zip.

```


```python
# Sample data
data = {'text': ["This is a good movie",
                 "This is a bad movie",
                 "I like this food",
                 "I dislike that food",
                 "good job",
                 "bad luck",
                 "I love this",
                 "I hate that",
                 "this is a masterpiece",
                 "this is not good"],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative','positive','negative']}
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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>This is a good movie</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is a bad movie</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I like this food</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I dislike that food</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>good job</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)

df['text'] = df['text'].apply(preprocess_text)
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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>good movi</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bad movi</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>like food</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dislik food</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>good job</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

```


```python
# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

```


```python
# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
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
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>MultinomialNB</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.naive_bayes.MultinomialNB.html">?<span>Documentation for MultinomialNB</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>MultinomialNB()</pre></div> </div></div></div></div>




```python
# Make predictions
y_pred = model.predict(X_test_vectorized)
```


```python
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification report: \n", report)
```
`Output`
```
    Accuracy: 0.3333333333333333
    Classification report: 
                   precision    recall  f1-score   support
    
        negative       0.00      0.00      0.00         2
        positive       0.33      1.00      0.50         1
    
        accuracy                           0.33         3
       macro avg       0.17      0.50      0.25         3
    weighted avg       0.11      0.33      0.17         3
   ``` 
    



```python
# Save the model
filename = 'naive_bayes_model.pkl'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(vectorizer, open("vectorizer.pkl", 'wb'))
```


```python
# Load the model
loaded_model = pickle.load(open(filename, 'rb'))
loaded_vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
new_text = ["this was great", "this was horrible"]
new_text_vectorized = loaded_vectorizer.transform(new_text)
print("Loaded model prediction: ", loaded_model.predict(new_text_vectorized))
```
`Output:`
```
    Loaded model prediction:  ['positive' 'positive']
   ``` 

**Explanation:**

*   **Accuracy:** The accuracy score is approximately 0.33, or 33%. This indicates that the model correctly classified only one of the three instances in the test set. The model is not performing very well, this can also be due to random train test split. Accuracy is calculated as (TP+TN)/(TP+TN+FP+FN)
    
*   **Classification Report:**
    
    *   **Precision:**
        
        *   For the negative class, precision is 0.00, which means that the model was unable to correctly predict any negative class. TP/(TP+FP)
            
        *   For the positive class, precision is 0.33 which is calculated as (1/3), meaning one third of predictions for positive class were actually correct. TP/(TP+FP)
            
    *   **Recall:**
        
        *   For the negative class, recall is 0.00. This means that the model was not able to correctly identify any of the actual negative instances. TP/(TP+FN)
            
        *   For the positive class, recall is 1.00, which means that the model correctly identified all of the actual positive instances. TP/(TP+FN)
            
    *   **F1-score:** The F1-score is 0.00 for the negative class and 0.50 for the positive class. The F1-score is the harmonic mean of precision and recall, and it is calculated as: 2\*(Precision\*Recall)/(Precision+Recall)
        
    *   **Support:** The support shows the number of actual occurrences of that class in the test dataset. In the test set there are 2 instances of negative and 1 instance of positive class.
        
    *   **accuracy:** This shows the accuracy of the model as a whole, and is equivalent to the value of the Accuracy output.
        
    *   **macro avg**: Macro average of precision, recall and f1-score.
        
    *   **weighted avg**: Weighted average of precision, recall, and f1-score, weighted by the support of each class.
        
*   **Pickle:** The model and vectorizer are saved using pickle.dump, and loaded using pickle.load.
    
*   **Loaded model prediction:** Output of the loaded model on new data points.

Post-Processing
---------------

*   **Feature Importance:** Although not as straightforward as in tree-based models, feature importance can be interpreted from the model's parameters (probabilities of features for different classes).
    
*   **AB Testing:** If you have multiple ways to preprocess text or numeric features, AB testing can be done to find which preprocessing gives the best model.
    
*   **Hypothesis Testing:** Statistical significance of the model performance can be tested using hypothesis tests.
    
*   Other statistical tests can also be used for post processing if needed.
    

Hyperparameter Tuning
---------------------

```python
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary nltk resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Sample data
data = {'text': ["This is a good movie",
                 "This is a bad movie",
                 "I like this food",
                 "I dislike that food",
                 "good job",
                 "bad luck",
                 "I love this",
                 "I hate that",
                 "this is a masterpiece",
                 "this is not good"],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative','positive','negative']}
df = pd.DataFrame(data)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)

df['text'] = df['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)


# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Define hyperparameter grid
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 3.0]}

# Grid search for hyperparameter tuning
grid = GridSearchCV(MultinomialNB(), param_grid, cv=2)
grid.fit(X_train_vectorized, y_train)
print(f"Best parameters for Naive Bayes: {grid.best_params_}")
print("Best Score for Naive Bayes: ", grid.best_score_)
```
`Output:`
```
Best parameters for Naive Bayes: {'alpha': 2.0}
Best Score for Naive Bayes:  0.25
Accuracy: 0.3333333333333333
Classification report: 
               precision    recall  f1-score   support

    negative       0.00      0.00      0.00         2
    positive       0.33      1.00      0.50         1

    accuracy                           0.33         3
   macro avg       0.17      0.50      0.25         3
weighted avg       0.11      0.33      0.17         3
```

Checking Model Accuracy
-----------------------

Model accuracy can be checked using the following metrics:

*   **Accuracy:** Fraction of correct predictions. (TP+TN)/(TP+TN+FP+FN)
    
*   **Precision:** Measures the proportion of true positives among all positive predictions. TP/(TP+FP)
    
*   **Recall (Sensitivity):** Measures the proportion of true positives among all actual positives. TP/(TP+FN)
    
*   **F1-Score:** Harmonic mean of precision and recall. 2\*(Precision\*Recall)/(Precision+Recall).
    
*   **AUC-ROC:** (Area Under the Curve - Receiver Operating Characteristic). This metric is useful for checking the overall model quality.
    
*   **Confusion Matrix:** A table that summarizes the performance of a classification model.
    

Productionizing Steps
---------------------

*   **Local Testing:** Create a test script to check that the model is loading correctly and giving the expected output.
    
*   **On-Prem:** Containerize the code and model using docker, and deploy on server.
    
*   **Cloud:** Deploy using cloud provider's platform. Use cloud services to train, test and deploy.
    
*   **Real time and Batch:** Setup an ingestion pipeline for real time and batch prediction.
    
*   **Monitoring:** Monitor the model output for any deviations.
    

Conclusion
----------

Naive Bayes is an efficient and effective algorithm, particularly when it comes to classifying text data. Its simplicity and speed make it a great choice when dealing with large text datasets.Despite the "naive" assumption of independence of features, the Naive Bayes algorithm is used effectively in various real world applications. However, more sophisticated models are replacing Naive Bayes, especially when dealing with numerical data. It is valuable for learning and understanding machine learning concepts.

References
----------

1.  **Scikit-learn Documentation:** [https://scikit-learn.org/stable/modules/naive\_bayes.html](https://www.google.com/url?sa=E&q=https://scikit-learn.org/stable/modules/naive_bayes.html)
    
2.  **NLTK Documentation:** [https://www.nltk.org/](https://www.google.com/url?sa=E&q=https://www.nltk.org/)
    
3.  **Wikipedia:** [https://en.wikipedia.org/wiki/Naive\_Bayes\_classifier](https://www.google.com/url?sa=E&q=https://en.wikipedia.org/wiki/Naive_Bayes_classifier)