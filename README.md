# Comparative Analysis of Sentiment Analysis Algorithms


Sentiment analysis is a challenging subject in machine learning. People express their emotions in language that is often obscured by sarcasm, ambiguity, and plays on words, all of which could be very misleading for both humans and computers

This tutorial is designed to give you a comparative analysis of different sentiment analysis algorithms in the increasing order of acuracy.

### Dataset Description

This dataset is collected from Hotel Booking Website. Each row has a hotel review named as **Description** with associated sentiment under **Is_Response** column. We will use this pre labeled dataset for training various machine learning models and based on the model we would predict the label for an unknown moview review. 

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Description</th>
      <th>Browser_Used</th>
      <th>Device_Used</th>
      <th>Is_Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id10326</td>
      <td>The room was kind of clean but had a VERY stro...</td>
      <td>Edge</td>
      <td>Mobile</td>
      <td>not happy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id10327</td>
      <td>I stayed at the Crown Plaza April -- - April -...</td>
      <td>Internet Explorer</td>
      <td>Mobile</td>
      <td>not happy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id10328</td>
      <td>I booked this hotel through Hotwire at the low...</td>
      <td>Mozilla</td>
      <td>Tablet</td>
      <td>not happy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id10329</td>
      <td>Stayed here with husband and sons on the way t...</td>
      <td>InternetExplorer</td>
      <td>Desktop</td>
      <td>happy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id10330</td>
      <td>My girlfriends and I stayed here to celebrate ...</td>
      <td>Edge</td>
      <td>Tablet</td>
      <td>not happy</td>
    </tr>
  </tbody>
</table>

## 1. Basic Natural Language Processing using Random Forest Classifier  [ Accuracy : 0.65293 ]


Random Forest Classifier works like a question & answer mechanism where every question has a binary answer. For e.g If a word 'Sad' is present in the review or not. A tree consisting of many questions like this yields to a decision like 'Good Review' or 'Bad Review'. Here each tree is a weak learner as it has only  set of questio. 

In random forest, we can combine decisions of many such weak learners to get a majority vote. For e.g if more than 50% trees says that the review is good than we can go with it.

###Getting started in python.

Here we will use **RandomForestClassifier** from **sklearn.ensemble** package. We need to clean our data in order to increase the acuracy. This step is called **Tokenisation**. Here **KaggleWord2VecUtility** provides the tokenisation function which removes the stop word and build a word vector from string having cleaner words. To count the frequency of each word in our dataset we will use **CountVectorizer** package.

```
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

```
Now import the training and testing dataset. Note that testing dataset contains only the description column, we will predict the class label for each row in testing dataset.

```
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

```

Initialize an empty list to hold the clean reviews

```
clean_train_reviews = []
```
Loop over each review; create an index i that goes from 0 to the length of the movie review list. This step will prepare a vector of words for each row in dataset which will be helpful in tokenisation step.


```
for i in range( 0, len(train["Description"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["Description"][i], True)))

```

Initialize the **CountVectorizer** object, which is **scikit-learn**'s bag of words tool. Here  **max_features = 5000** limits the model to prepare a dictionary of 5000 most frequent words in model. Otherwise we would get a very long vector to deal with. 

**fit_transform()** does two functions: First, it fits the model and learns the vocabulary; second, it transforms our training data into feature vectors. The input to fit_transform should be a list of strings.

**Numpy** arrays are easy to work with, so convert the result to an array.

```
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
                             
train_data_features = vectorizer.fit_transform(clean_train_reviews)

np.asarray(train_data_features)
                             
``` 
Initialize a Random Forest classifier with **100** trees as we discussed above. Fit the forest to the training set, using the bag of words as features and the sentiment labels as the response variable.

```
forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit( train_data_features, train["Is_Response"] )

```
Apply same treatment to our testing dataset and prepare a numpy array having word vectors.

```
clean_test_reviews = []

for i in range(0,len(test["Description"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["Description"][i], True)))
        
test_data_features = vectorizer.transform(clean_test_reviews)
np.asarray(test_data_features)    

```

Use the random forest to make sentiment label predictions. 
Copy the results to a pandas dataframe with an **"User_ID"** column and a **"Is_Response"** column. Use pandas to write the comma-separated output file.

```
result = forest.predict(test_data_features)

output = pd.DataFrame( data={"User_ID":test["User_ID"], "Is_Response":result} )

output.to_csv('first_sub.csv', index=False, quoting=3, header=True, columns=['User_ID','Is_Response'])

```

## 2. Gaussian Naive Bayse  [ Accuracy : 0.85 ]

Most of you have already studied the applications of Naive Bayse theorem  in Machine Learning for classification problem. Here we try to predict the probability of given review to be classified in each of the target_classes and assigned the label having heighest probability.

To implement Naivye Bayse Classifier, we will represent the review in feature vector form. Where we considet most frequent 500 words as features and count the number of occurences of each feature in given review. 

We can represent the review in simple **Word Count Vector Form** and also in **Weighted Vector From** form also known as the **Tf-Idf** or **Normalised Vector Representation**. Normalised vector are more reliable in the case where some words are very rare in vocabulary but which are also very important in deciding the class label. 

###Word Count Vector Representation

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col0</th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
      <th>col6</th>
      <th>col7</th>
      <th>...</th>
          <th>col495</th>
      <th>col496</th>
      <th>col497</th>
      <th>col498</th>
      <th>col499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
           <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.190959</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
            <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
      
      <td>0.0</td>
      <td>0.158427</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.4717</td>
      <td>0.0</td>
      <td>...</td>
            <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>...</td>
          <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>

### Tf - IDf Vector Representation

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
     
      <th>col0</th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
      <th>col6</th>
      <th>col7</th>
      <th>...</th>
      
      <th>col495</th>
      <th>col496</th>
      <th>col497</th>
      <th>col498</th>
      <th>col499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
            <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
     
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
     
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
            <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
 
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

###Getting started in python.

Here we will be using **GaussianNB** package of **sklearn.naive_bayes** for training the model and try to predict the class labels of unknown reviews. **LabelEncoder** is helpful in encoding the class labels to numeric value starting from **0** to **n_classes-1**. 

```
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer

```
Load data in to pandas dataframe.

```

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

```
**cleanData(text)** function extracts the words from reviews which may include html tags initially.

```
def cleanData(text):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]',r'',txt)
    txt = re.sub(r'\n',r' ',txt)

    return txt
    
``` 
Now we can concatinate the rows of test dataframe to train dataframe in order to clean data.

```
test['Is_Response'] = np.nan
alldata = pd.concat([train, test]).reset_index(drop=True)

```  

Clean **`Description`**.

```
alldata['Description'] = alldata['Description'].map(lambda x: cleanData(x))
```

Now initialise the functions - we'll create separate models for each type.

* **analyzer** : Whether the feature should be made of a word or character n-grams.
* **ngram_range** : The lower ans upper boundry of the range of n-values for different n-grams to be extracted.
* **min_df** : Maximum document frequency for given term to be selected as feature. Mostly stop words have df more than **min_df** are ignored here.
* **max_features** : Maximum length of the feature vector to restrict training time.

```
countvec = CountVectorizer(analyzer='word', ngram_range = (1,1), min_df=150, max_features=500)
tfidfvec = TfidfVectorizer(analyzer='word', ngram_range = (1,1), min_df = 150, max_features=500)

bagofwords = countvec.fit_transform(alldata['Description'])
tfidfdata = tfidfvec.fit_transform(alldata['Description'])

```

Encode the labels having categorical features in data given using **LabelEncoder()**.

```
cols = ['Browser_Used','Device_Used']

for x in cols:
    lbl = LabelEncoder()
    alldata[x] = lbl.fit_transform(alldata[x])
```
Create dataframe for features and set column names. Create separate data frame for bag of words and tf-idf. We will use both dataframes to predict labels write it to separate files.

```
bow_df = pd.DataFrame(bagofwords.todense())
tfidf_df = pd.DataFrame(tfidfdata.todense())

bow_df.columns = ['col'+ str(x) for x in bow_df.columns]
tfidf_df.columns = ['col' + str(x) for x in tfidf_df.columns]

bow_df_train = bow_df[:len(train)]
bow_df_test = bow_df[len(train):]

tfid_df_train = tfidf_df[:len(train)]
tfid_df_test = tfidf_df[len(train):]

```
Since we have already merged the training and testing datasets, now we have to separate both out. For that we can test if **Is_Response** column is null or not. If it is null than it belongs to test data.

```
train_feats = alldata[~pd.isnull(alldata.Is_Response)]
test_feats = alldata[pd.isnull(alldata.Is_Response)]
```
Set target variables to **1** if **happy** and **0** if **sad** so its easy matematically to compute class label. Append feature vectors to training and testing datasets.

```
train_feats['Is_Response'] = [1 if x == 'happy' else 0 for x in train_feats['Is_Response']]

train_feats1 = pd.concat([train_feats[cols], bow_df_train], axis = 1)
test_feats1 = pd.concat([test_feats[cols], bow_df_test], axis=1)

test_feats1.reset_index(drop=True, inplace=True)

train_feats2 = pd.concat([train_feats[cols], tfid_df_train], axis=1)
test_feats2 = pd.concat([test_feats[cols], tfid_df_test], axis=1)

```
Now let us predict the unknown class labels using **GaussianNB( )**. 

```
mod1 = GaussianNB()
target = train_feats['Is_Response']

clf1 = GaussianNB()
clf1.fit(train_feats1, target)

clf2 = GaussianNB()
clf2.fit(train_feats2, target)

preds1 = clf1.predict(test_feats1)
preds2 = clf2.predict(test_feats2)

```
Since model will predict the numerical value assigned to the class either 0 or 1. We need to convert it back for us to understand and then write it to csv files. 

```
def to_labels(x):
    if x == 1:
        return "happy"
    return "not_happy"
    
sub1 = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':preds1})
sub1['Is_Response'] = sub1['Is_Response'].map(lambda x: to_labels(x))

sub2 = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':preds2})
sub2['Is_Response'] = sub2['Is_Response'].map(lambda x: to_labels(x))

sub1 = sub1[['User_ID', 'Is_Response']]
sub2 = sub2[['User_ID', 'Is_Response']]

sub1.to_csv('submissions/sub1_cv.csv', index=False)
sub2.to_csv('submissions/sub2_tf.csv', index=False)

```    
