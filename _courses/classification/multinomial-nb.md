---
title: "Decoding Text: A Deep Dive into Multinomial Naive Bayes"
excerpt: "Multinomial Naive Bayes Algorithm"
# permalink: /courses/classification/multinomial-nb/
last_modified_at: 2022-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Machine Learning
  - Text Classification
tags: 
  - Machine Learning
  - Classification Model
---


{% include download file="multinomial_nb.ipynb" alt="Download Multinomial Naive Bayes Code" text="Download Code" %}

## Introduction: Making Sense of Words

Imagine you want a computer program that automatically sorts through a pile of documents or classify emails or categorize customer reviews as positive or negative automatically. Text classification is precisely that - the process of making a computer automatically put pieces of text into predefined categories.  **Multinomial Naive Bayes** algorithm provides us one of the easiest to understand techniques to classify texts, it gives pretty reasonable results and does not require expensive hardware to train unlike many modern techniques that performs similarly in many practical use cases. 

In simple language, this is like having a very helpful robot that categorizes different pieces of text, from emails, social media texts, to any large documents that is automatically classified by computers.

**Examples of Use**
* **Spam Email detection**: Automatically detecting spam and moving them into your spam folder
*   **Review Classification:** Classifying whether product reviews or service feedbacks are *positive* or *negative*, or even *neutral*, for customer service and overall service experience improvement
*  **News Article Category Prediction:** Tagging of news as whether belonging to  *sports*, *politics* or *entertainment*, etc
* **Complaint Tickets Tagging:** Classifying tickets as technical issue or sales, so that automated processing with respect to urgency and different team departments, thus providing better organizational process.

Let's go through all the mathematics behind the simple algorithm, it might sound complex, but after understanding all the logic it will be much simpler and clearer

## Math of Multinomial Naive Bayes

Multinomial Naive Bayes algorithm heavily relies on the concept called **Bayes' Theorem**. Bayes theorem tells us how to calculate the likelihood of an event using knowledge or historical data of a related event.  Mathematically, the formula appears like this:

$$
P(A|B) = \frac{P(B|A) * P(A)}{P(B)}
$$

In simpler terms it means: The probability of 'A' happening if 'B' already occurred, i.e.  `P(A|B)` can be derived via computing probabilities of related other cases `P(B|A)`,  `P(A)` and  `P(B)`. Let us explain more

* `P(A|B)`: This reads as the **posterior probability** . The word *posterior* is associated when an observation or the event has already happened and its derived using a priori distribution. This what we really want, probability of event 'A' if the event 'B' is already known or observed
*   `P(B|A)`:  this is known as **likelihood probability**, how probable the event 'B' happening if the event A occurred.
*   `P(A)` :  this the probability we would give without considering anything prior observations and is called the **prior probability**
*   `P(B)`: this term only normalizes result and represents that event `B` occurred during historical time

**Text Classification**:
Assume *c* represents text categories (e.g., 'spam', 'not spam'), and *d* denotes a document. Our aim is: calculate probability that "Document *d*  belongs to class/category *c*" represented by term  $$P(c|d)$$. With a slightly different re-arrangement of Bayes theorem

$$
P(c|d) = \frac{P(d|c) * P(c)}{P(d)}
$$

In this re arrangement terms:
*   `P(c|d)`  means : "what is the probability of belonging to class *c* , given I saw this document *d*?". It is  called the posterior probability, since it denotes after the event we derived new class probability. It is also our prediction goal. 
* `P(d|c)`: This means the "probability of seeing this document *d*, if it already belongs to *c*" (likelihood). In practical terms for texts we can use "the frequency with which we saw this type of document in category `c`.
* `P(c)`: This the the proportion of training examples falling under a label 'c'. For Example total count of spam in dataset by total counts of emails from training dataset
* `P(d)` : The term is often just seen for normalizing purposes during training phase. It is used to scale and adjust output during the application of classifier

**The "Naive" Assumption:**

A "Naive Bayes classifier" assumes every words are independent of each other within category of classes given documents, hence likelihood $$ P(d|c) $$, 
where a document `d` which consists of word set $$ (w_1, w_2,..., w_n) $$ where `n` are the words in document is given as the product

$$ P(d|c) = P(w_1|c) * P(w_2|c) * ... * P(w_n|c) =  \prod_{i=1}^n P(w_i|c)  $$

where $$P(w_i | c)$$ denotes probability of seeing a word $$w_i$$ when the document actually belongs to class $$c$$. This term is the likelihood term mentioned earlier which must be determined
We use training data to find,

$$ P(w_i | c) = \frac{\text{count of word } w_i \text{ in category } c + 1}{\text{Total words in category } c \text{ + Number of unique words in the dataset}} $$

We add one smoothing value (laplace smoothing factor or adding one),  this avoid zero counts and handles unseeen text during production predictions by assigning probability rather than skipping the calculations by making total output equal to zero.
  The word "Naive" comes from assuming that the all of the above calculations are derived under that every word is independent within each other in a given category . This can often violate and can have huge dependencies in sentence but is often robust in real application scenarios with large texts documents to get usable and valuable predictions.


**Example:**

Let's say our classifier has to determine spam emails with the words given below in training dataset, based on word counts for class spam and not spam

|       |     "free"     |    "money"   |   "offer"  |   "report" |  "meeting"   |  "project"   |
|-----------|--------------|-------------|-----------------|--------------------|---------------|----------|
| spam      |      5    |     4    |     3      |    0           |     0    |  0          |
| not spam   |     0        |       0   |        0   |      3          |     5       |     4     |

Assume we observe during actual classification  a new *document* : 'free money project', we want to compute

*   `P(document is spam  | 'free money project')`, using our computed word statistics or likelihood counts from model trained using historical documents for class 'spam' and 'not spam', from table shown before
*   `P(document is not spam  | 'free money project')`. We make a choice by selecting most probable among these 2 choices from Naive Bayes Model (the class type that is having most probability value with that category).
Using  data above, lets calculate probability of individual words first.

$$
P('free' | \text{spam}) = \frac{5 + 1}{5 + 4+3 +6} = 6/18
$$

similarly,

$$
P('money' | \text{spam}) = \frac{4 + 1}{5 + 4+3 +6} = 5/18
$$

$$
P('project' | \text{spam}) = \frac{0 + 1}{5 + 4+3 +6} = 1/18
$$

Note that, the term '6' at bottom comes from the unique number of vocabulary in whole dataset which is = (free, money, offer, report, meeting, project) total vocab size = 6
And, if *n* represent, a new text "free money project" and spam class is called `s`,

$$ P( \text{new text } n | \text{spam category } s )  =  \frac{6}{18} \times \frac{5}{18} \times \frac{1}{18} $$

And also:

$$ P('free' | \text{not spam}) = \frac{0 + 1}{3+5+4+6} = 1/18 $$

$$ P('money' | \text{not spam}) = \frac{0 + 1}{3+5+4+6} = 1/18 $$

$$ P('project' | \text{not spam}) = \frac{4 + 1}{3+5+4+6} = 5/18 $$

  Similarly,

$$ P( \text{new text } n | \text{not spam category})  =  \frac{1}{18} \times \frac{1}{18} \times \frac{5}{18} $$

Using historical frequencies and the total spam emails in our corpus we will use calculate, say
`P("spam") = 0.30`, the actual value comes by dividing the total frequency of all training dataset belonging to a category. Therefore we can assign
 `P("not spam") = 0.70`
 We simply do product multiplication from the individual terms from Bayes Theorem as shown. Usually logarithmic product form is often used as it handles better underflow problem of floating point representation of numbers on computer due to multiple probabilities often ranging between 0 to 1

So given the equations for the *Posterior Probability*, `P(category c | document d)`,  where category belongs to *spam* or *not spam*. we select and assign that category that has the maximum probability using results from above individual calculation based on product rules.

## Preparing Data: Necessary Steps and Assumptions

For Multinomial Naive Bayes to be correctly trained and useful in applications the underlying data must follow certain guidelines
**Assumptions:**

* **Conditional Independence:**  (discussed above in Math Section), All terms, (usually single terms, pairs of terms , triples of terms, bigram trigram and n-gram ), these should be all mutually statistically independent from each other given a text documents classification
* **Count based feature data:** All input must be of positive numerical form to compute count of terms, example being frequency of how many times words exist in the documents during training phase, rather than scaled values (normalized), such as decimal value (where mean is 0, with deviation one, or scaled between range zero and 1 ) which distorts count
*  **Discrete Categories:** Model must output categories as labels (is a text 'sports' or 'not sports'), we cannot have regression with predicted label as an real continuous floating values using the standard form of Multinomial Naive Bayes Algorithm (as other regressional techniques such as liner regressions). So model labels needs to discrete, finite numbers for example: Spam, Not spam or Positive, Negative.

**Checking Assumptions**
The independence is often the major assumption that may cause issue in data. A drastic poor performance usually hints violations to these underlying model assumptions during train test classification evaluation metrics. So this is indirectly inferred (from poorer overall model outputs or through visual investigation ). Increasing n grams can weaken this assumption and also will improve performance to model accuracy. More complicated Deep learning models are the often preferred, alternative choice for problems where conditional dependence assumptions for bag-of words may drastically lower the model prediction outcomes, and must be considered given time resources or application constraints, these models learn arbitrary dependencies.

**Required Libraries (Python)**:

*   **scikit-learn (sklearn):** This is a foundational python module for performing almost all core machine learning functionality in single module and has robust tested implemenation of almost all core traditional statistical and machine learning algorithms (this can perform tokenization text extraction numerical transformaitons using many other modules such as scikit feature extract). For our particular application we use MultinomialNB.
*   **pandas:** This data module allows storage of input/outputs and transformations for a wide range of tables dataset structure similar to data in spreadsheets. This makes it user friendly to perform common numerical and other format type tasks on real datasets for exploration
* **numpy:** Core numerical data structure useful in perform operations. (most mathematical operations, statistical calculations, array transformation, etc can be performed effectively.)

 Install all by this command  `pip install scikit-learn pandas numpy`

##  Data Preprocessing

Prior to feeding your documents into our  algorithm data requires specific types of processing,
*   **Numerical transformations:** All our inputs needs to be in numeric, but count like in forms
*  **No Data Normalization Required**:  Do NOT scale the features to have zero mean or values between range (0, 1). Standard normalization method often deteriorates result in Naive Bayes Algorithm which needs count value for deriving probabilistic representation. MultinomialNB models perform badly if data scaling/normalization is applied. Hence only positive value numeric counts representing word occurrence must be used to have correct probabilities of words or N-gram sequence given specific class categories for document analysis task during both training phase and production (predict) phase for the trained classifier.

**Text Preprocessing Steps**: 

*  **Lowercasing**: The model learns the lowercase and cannot identify *Word* different from the *word*. All similar words like "The" to "the" will help model accuracy when transformed by `lower()`
* **Removal of Punctuation**: `!,."#$` characters should not add valuable features for a classifying model and should be usually removed in all models.
* **Tokenization**: All sentence like text should be divided to token or unit usually via the `split` function to convert *This is text.* -> to tokens [This, is, text, . ] which are easier to process, prior to counting frequency with respective category, each words are usually called tokens in such context
*   **Stop Words**: Words such as *the, is, a,* are so common they barely contribute any insight when computing different features of specific text and must be filtered prior to feature (word/n gram count creation.)
*   **Stemming/Lemmatization**: words with same root format can be derived and consolidated using these transformation in order to correctly make use of counts/frequencies during model parameter learning, so that variability of single words (*running, run*)  with respect to same or close semantic are counted effectively
   
**When to ignore it?**
Small data where visual classification or any similar approaches can also do better by eyeballing does not really require implementation and usage of text based models including this algorithm. Such approaches may work on those circumstances, and we might skip implementation entirely. And all pre processing usually only adds value by the generalizable model. Pre processing is mainly implemented for ensuring correct generalized model (as for different dataset text could have difference formats which is tackled during all text preprocessing phases as mentioned before). Hence you usually want some pre processing (but some specific normalization type processing steps must not be implemented for Naive Bayes algorithms that perform operations with numeric counts of word tokens), if model needs to be generalized

**Examples:**
   Original sentence such as *"The weather is good !!!"*  is converted in sequential preprocessing

* Lowercasing ->  *"the weather is good !!!"*
* Punctuation removal -> *"the weather is good "*
* Stop-word Removal -> *"weather good"*
* Tokenization (or Split function on string )-> ["weather", "good"] which can be given to algorithm
* Stemming /Lemmatization: "running", "ran", "runs" become, "run" usually

 These are finally the most common and most effective processing in natural language texts as applied prior to using machine learning (but must remember *count based transformation must always come prior to usage, not the standard Normalization, as required by the Naive Bayes model's assumptions*. We cannot scale values such that `mean =0` or that all values lie in between `[0, 1]` like other algorithms.). We always must apply numerical *count/frequency of N gram words*, such as pair/triple combinations as final feature in our text data, then this integer representation from training dataset will now be ready for input for multinomial Naive Bayes Training and evaluation using python

## Practical Implementation using Python

Let's put the theory to work in a code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# Dataset is the real data that has some texts that requires categories of text assigned ( labels) and is supplied usually as input to any machine learning algorithm in real scenarios.
data = {
    'text': [
        "This is a really good python programming book",
        "You can easily learn data science using this book, python based",
        "Technologies based on python are very interesting and make my days interesting.",
        "This was really very awful useless python book",
         "What a great device!! I really appreciate its feature set, wow",
        "I am going to play football today.",
        "The match today is amazing, everyone loves it, great.",
        "The hike today was sunny",
        "what a nice wonderful hike with my best friends!",

    ],
    'category': [
        "programming",
        "programming",
        "programming",
        "programming",
       "tech",
       "sports",
        "sports",
        "outdoor",
         "outdoor"
        ]
}
df = pd.DataFrame(data)


# data into train (80%) and test split(20%) - so that there are separate datasets that are used during final accuracy check with data never seen during the learning process, ensuring generalization to test accuracy
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)

# count vectorizer used for numerical (frequency/counts) of all words from our train data, it contains feature vocabulary too derived during fit operation. After it also returns numeric sparse count of our vocabulary during fit and also when we transform via count transform method call.
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initiate the naive bayes classifier here after converting text into correct format with `CountVectorizer()` object instance above.
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Model is ready to perform prediction from any unseen datasets
y_pred = model.predict(X_test_vectorized)


# computes the output for understanding results by printing precision and recall. and F1
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Saves our trained model and countvectorizer so that we don't need to retrain for reuse or deploy for production (web, or service end points for making classification using real world texts that requires class/category assignment
joblib.dump(model, 'multinomial_nb_model.pkl')
joblib.dump(vectorizer, 'count_vectorizer.pkl')


# Loads the above saved object to use from stored file, this way if someone used our saved data (text classifers) and feature extractor data they do not have retrain whole pipelines again
loaded_model = joblib.load('multinomial_nb_model.pkl')
loaded_vectorizer = joblib.load('count_vectorizer.pkl')


# Test the saved data objects by performing class predictions on real or synthetic newly text samples/document sets by real end-user applications
new_data = ["Machine learning can really empower tech users to solve tough issues for business", "Football match today was a bit frustrating","what a perfect sun and beach"]
new_data_vectorized = loaded_vectorizer.transform(new_data)
new_predictions = loaded_model.predict(new_data_vectorized)

print(f"Class predictions of new data sets : {new_predictions}")

```

**Code Breakdown**

* Import Libraries: import needed modules pandas, scikit-learn feature_extraction and naive bayes module and other metrics to test accuracy, numpy (and module joblib). These libraries allow easy programming without worrying low level implementation details for the naive bayes implementation from underlying models

* Dataset Creation: It contains dictionary for text as the text corpus, and its actual label names as category. Pandas module helps us read it with tabular column data format. The purpose is to simply read some examples. (real use cases uses excel file , or database to perform same read methods and also includes significantly large document sets often)

* Train and Test Split: Creates training and testing splits for generalizing models by sklearn utility with text column and label columns.

* Text vectorizing: This CountVectorizer transforms text column using its fit_transform methods in both test train, where vocabulary of all words from train text is stored, then transform from that method return our numerical transformation of texts by counts to feed our Naive Bayes algorithm implementation. In our case feature set is word and this produces sparse matrix format. The numerical sparse matrices (from vectorizer), represents how frequent all unique training words from dataset is actually present in training or test documents for classifier input.

* Train: Train and fit are different operations here for Scikit learn implementations, where MultinomialNB method uses word counts of training, labels. Now we are ready to do text classification.

* Prediction and Evaluation: Predictions for all test sets now generates its predicted classes from models using training phases information and then sklearn classification report (this report accuracy precision recall etc) , to evaluate quality for that particular test. The actual result values of actual (labels are given during splitting and is also given for each test sample document to sklearn utilities by function classification_report). These test metrics tell whether the fitted models are sufficiently performant

* Saving models: These lines use Joblib modules from python, this helps serializes Python models/ objects like classifer objects and our vectorizer class to make is easy to perform inference and classification predictions after reloads and to avoid retraining, these files can now also be supplied with software as long the python implementation environment where the trained saved files can be reloaded and also utilized for predictions. It uses pickling object mechanisms, but in better more flexible ways by serializing the entire classifier object or text feature transformers in a simple convenient manner to perform easy integration and use across software

* Loading objects: Load models are now imported back by reverse of method where stored saved method was used before

* Make real classifications with new data: Finally after reloading the model and feature transformer objects we make classifications by text prediction to verify implementation worked well, or show that with newly created example sentences with unseen class label information during previous steps

`Output:`

The code when executed gives a text based printed tabular output and predicted class.

```
Accuracy: 0.3333333333333333
              precision    recall  f1-score   support

       outdoor       0.00      0.00      0.00         1
     programming       0.50      1.00      0.67         1
        sports       0.00      0.00      0.00         1

       accuracy                           0.33         3
      macro avg       0.17      0.33      0.22         3
   weighted avg       0.17      0.33      0.22         3

Class predictions of new data sets : ['programming' 'sports' 'outdoor']
```
**The printed outputs includes**

The evaluation scores computed with sklearn (precision recall and f1 score with train data sets with both averaged scores both by macro (averages), or also average given frequency weighting for imbalanced datasets) where you see output, with support giving frequency of number examples by category available during testing split

And you get the category or labels output with model output showing programming, sports , and outdoor respectively as an example test run on real output results of your implemented program shown here (Note results may differ if different sets of sentences with more texts/data examples, the important to notice all predicted and labelled should align reasonably using different inputs for large/robust scenarios.)
In simple terms these two segments shows overall ability of your program for text categorization on given texts or inputs that is used.

## Post-Processing

Although feature importance, of individual words cannot be simply identified for naive Bayes without deriving, following techniques is employed for further improvement/analysis.

* Likelihood based feature:
Extracting feature_log_prob_ can show words that have greater effect towards specific categories (high log means larger probabilities) which also helps get intuitions about text document and categories used.

* A/B Tests When performing tests using new sets of classifier and with original ones will identify, or help us decide whether model change improves performance significantly, by checking if A / B is really performing significant better than the baseline

* Statistical/Hypothesis tests: This allows user to do further testing on important variables derived for performance/ model interpretations. For Example whether correlation for a specific words to labels have a statistical difference for label classifications to specific target.

## Hyperparameters and Parameter Tuning

Our Multinomial Naive Bayes class from sklearn mainly only has one significant tuning knob (hyper parameter setting ) to adjust alpha

* **alpha**: This is referred to as Laplace smoothing or additive smoothing constant (adding 1 by convention is called laplace and less than or higher are called general additive Lidstone method). During the probabilistic calculation on words count or the words features given some target label or category c, we add this extra 1 to count, so during testing phase (or evaluation time) with a new sentences / documents with unseen word is handled with non zero probability values rather than completely being ignored making classification output 0 (i.e making it totally inaccurate by discarding unseen vocabulary of a new text that has not encountered in dataset during training stage ). Very high values make features same distributions of probablity regardless of any inputs while non zero values makes handling unforeseen inputs relatively more robust. The sklearn uses parameter named alpha which performs smoothing method, its better set non-zero small value say [0.01 , 0.1, 1 ], this improves over all test metric accuracies and avoid overfits.

Here's python code to use Grid Search Cross Validation, it computes best optimal values (optimal performance value of cross validation), that provides robustness

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.01, 0.1, 1, 2, 5, 10]}

# set of  paramters
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring = 'accuracy')

grid_search.fit(X_train_vectorized, y_train)
best_alpha = grid_search.best_params_['alpha']

# Uses the cross-validation object of scikit-learn
print("best parameter ",best_alpha)
best_model = MultinomialNB(alpha = best_alpha)
best_model.fit(X_train_vectorized, y_train)

y_pred = best_model.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))
```

The Grid search finds best value with training data to give good test results. A best performing alpha then passed during initial training step.

Evaluating Performance

* **Accuracy**:
How many results were classified with label given using predicted method is correctly classified compared to all labelled from test document

`𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦 = Number of Correct Predictions/Total Number of Predictions`

* **Precision**: The ratio of correctly labeled positive text/docs vs all predictions where algorithm also predicted true class categories from classifier

`Precision=True positives+False positives/True positives`

* **Recall**: Number of positive values divided by number of real/ground truths available in a test case with labeled examples.

`Recall=True positives+False negatives/True positives`

* **F1-score**: It's the most practical overall score for most applications to judge classifiers performance, combines the precision and recall to a number. A higher value means more reliable method to label correctly with overall more correct positives

`F1-score=2×Precision+Recall/Precision×Recall`
	​

Classification report in sklearn displays all this scores by providing clear insights about our algorithms results, using labels of true (actual labelled class from split or separate set during production stage) or the classes produced as the final classification using multinomial Naive Bayes

## Deployment in Practice

Here's brief procedure for deployment into any system, cloud or server infrastructure. The trained objects also will need to bundled in an independent, and specific method, using code implementations

Proper Model Validation & testing: Evaluate via multiple train test runs, validation approaches and ensure model performs acceptably. In most software setup, these can involve real user text from similar distributions to ensure robust overall results

Store the fitted objects: The fitted objects model vectorizer from above code must be properly stored in file location for real world software/systems deployment.

Web deployment / Service endpoints expose model for real prediction tasks over web servers in an interface.
Here python based flask for API exposure is usually a common solution, we demonstrate minimal steps using single service function below which assumes vectorizer/model were given from specific method from server:

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Provide code/methods to initialize models
model = joblib.load('multinomial_nb_model.pkl') # usually this must provided from location specific on deployment systems/platforms.
vectorizer = joblib.load('count_vectorizer.pkl')# # usually this must provided from location specific on deployment systems/platforms.

@app.route('/predict', methods=['POST'])
def predict():
   data = request.get_json()
   text = data.get('text')
   if not text:
       return jsonify({'error': 'No text available for classification!'}), 400
   # get counts from the incoming data that you wish to label
   text_vectorized = vectorizer.transform([text])
   prediction = model.predict(text_vectorized)[0]  # first output selection since prediction result return array-like types
   return jsonify({'prediction': prediction})

if __name__ == '__main__':
       app.run(debug=True, port = 5000)
```

These few lines provides basic methods where client apps can send text in json format, it produces corresponding prediction label via trained models and their encoding data, all via web requests from client (other software, browser applications )
4. Model Monitoring - Continually test, improve via re training pipelines.

Collect model logs/ classification result into datasets, to be re trained or fine tune if performance dips down in practice (this often uses re running data engineering methods on production level, before another model and pipelines will be re deployed).

If new categories are introduced for the systems that needs to support those extra data classes during real user input on client/servers during productions systems, full steps, will be involved including creating dataset, re-run cross-validation, saving objects, API interface integration and monitoring.
The methods mentioned above applies to cloud, on-premise and other general cases for real usage applications

## Conclusion

Multinomial Naive Bayes classifier with text as feature is widely implemented still now due to ease, quick to deploy (both to develop/ and put to software production phases compared to complex deep learning alternatives. It provides effective results using correct parameters values using simple intuitions about underlying probabilities and frequencies on training datasets for real document and text categories (labels ). And they are still extremely valuable tools, mainly for being easily interpretable given basic explanations by simply computing count/word based model rather then many difficult black boxes machine learning model counterparts in many complex AI architectures used in most practical real systems nowadays.

## References

* Scikit-learn's Naive Bayes Guide: https://scikit-learn.org/stable/modules/naive_bayes.html

* Wikipedia Documentation for Naive Bayes method : https://en.wikipedia.org/wiki/Naive_Bayes_classifier

* Text Mining Class Material using text classification: https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf

* University of Washington course data : https://courses.cs.washington.edu/courses/cse599c1/22au/files/1-2-naive_bayes.pdf
