#!/usr/bin/env python
# coding: utf-8

# In[1]:


##importing required libraries
# import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# <h1>Functions</h1>

# In[2]:
if len(sys.argv) < 3:
    print("Enter complete arguments")
    print("#1 Enter database")
    print("#2 Address to save pickle file")
    quit()


#function to remove punctuation or any special characters
def removePunc(sentence): 
    processed = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    processed = re.sub(r'[.|,|)|(|\|/]',r' ',processed)
    processed = processed.strip()
    processed = processed.replace("\n"," ")
    return processed


# In[3]:


def removeNonAlphabets(sentence):
    final_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        final_sent += alpha_word
        final_sent += " "
    final_sent = final_sent.strip()
    return final_sent


# In[4]:


def removeHTMLtags(sentence):
    tags = re.compile('<.*?>')
    cleaned = re.sub(tags, ' ', str(sentence))
    return cleaned


# In[5]:


def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


# In[6]:


##use SnowBall Stemmer for Stemming purpose
stemmer = SnowballStemmer("english")

def stemming(sentence):
    finalSentence = ""
    for word in sentence.split():
        stemmed = stemmer.stem(word)
        finalSentence += stemmed
        finalSentence += " "
    finalSentence = finalSentence.strip()
    return finalSentence


# In[7]:


def clean_text(text_data):
    text_data = text_data.apply(removeHTMLtags)
    text_data = text_data.apply(removePunc)
    text_data = text_data.apply(removeNonAlphabets)
    text_data = text_data.apply(removeStopWords)
    text_data = text_data.apply(stemming)
    return text_data


# In[8]:


def read_database(filename):
    '''
        Function reads a SQLite Database File and converts it to a DataFrame
    '''
    # Creating SQLite Engine to import DataFrame as DataBase
    #engine = create_engine('sqlite:///DisasterResponse.db')
    engine = create_engine(f'sqlite:///{sys.argv[1]}')
    data = pd.read_sql_table(table_name = 'InsertTableName', con = engine)
    return data


# In[9]:


def vector_generator(train_df, test_df, analyzer_, stop_words_, max_features_, labels_to_drop):
    '''
        Function takes in training and testing data and generates and returns feature vectors
    '''
    vectorizer = TfidfVectorizer(analyzer = analyzer_ , stop_words = stop_words_, max_features = max_features_)

    train_text = train_df["message"]
    test_text = test_df["message"]

    vectorizer.fit(train_text)
    vectorizer.fit(test_text)

    x_train = vectorizer.fit_transform(train_text)
    y_train = train_df.drop(labels = labels_to_drop, axis=1)

    x_test = vectorizer.fit_transform(test_text)
    y_test = test_df.drop(labels =  labels_to_drop, axis=1)
    
    return x_train, y_train, x_test, y_test


# In[10]:


def generate_results(y_test_array, predictions, fields):
    '''
        Function accepts test data output, predictions from model, and fields and returns classification report 
    '''
    return classification_report(y_test_array, np.array([x[:] for x in predictions]), target_names = fields)


# In[11]:


def best_estimator(training_x, training_y, model_parameters, cross_validation_count):
    '''
        Function takes in training data and model_parameters to be tested, 
        returns bests parameters found using GridSearch
    '''
    # defining parameter range
    researched_grid = GridSearchCV(RandomForestClassifier(), model_parameters, cv=cross_validation_count, scoring='accuracy', return_train_score=False, verbose=1)

      # fitting the model for grid search
    grid_research=researched_grid.fit(training_x, training_y)

    # getting the best estimator
    best_model = grid_research.best_estimator_ 
    
    return best_model


# <h1>Deployment using Random Forest CLassifier and TF-IDF</h1>

# In[12]:

##FOr
if __name__ == "__main__":
    # Downloading stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)##compiling into regex to remove them

    ##use SnowBall Stemmer for Stemming purpose
    stemmer = SnowballStemmer("english")
    
    df = read_database('sqlite:///DisasterResponse.db')
    print(df.head())
    
    df.drop_duplicates(subset = "id", inplace=True)
    
    df["message"] = clean_text(df["message"])
    # replaces anomalous values (2) in data with 1
    df["related"] = df["related"].map(lambda a: 1 if a > 1 else 0)
    df.to_csv("../Cleaned.csv")
    print(df.head())
    
    train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)

    # Feature Vectors are generated
    x_train, y_train, x_test, y_test = vector_generator(train, test, "word", "english", 500, ['message','id','original','genre'])

    #RandomForestClassifier is used to build model
    clf2 = MultiOutputClassifier(RandomForestClassifier()).fit(x_train, y_train)
    predicted = clf2.predict(x_test)
    
    print(generate_results(y_test.iloc[:, :].values, predicted, y_test.columns))
    pickle.dump(clf2, open(f'{sys.argv[2]}','wb'))


# # # Using bag of words feature extractor and random forest classifier(best performance on targeted data)

# # In[13]:


# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(max_features=500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# X_train = vectorizer.fit_transform(train["message"]).toarray()
# X_test = vectorizer.fit_transform(test["message"]).toarray()


# # In[14]:


# X_train.shape


# # In[15]:


# Y_train = train.drop(["id", "message", "original", "genre"], axis = 1)


# # In[16]:


# Y_test = test.drop(["id", "message", "original", "genre"], axis = 1)


# # In[17]:


# Y_train.shape


# # In[18]:


# # RandomForestClassifier is used to build model
# clf3 = MultiOutputClassifier(RandomForestClassifier()).fit(X_train, Y_train)
# predicted = clf3.predict(X_test)


# # In[19]:


# print(generate_results(Y_test.iloc[:, :].values, predicted, Y_test.columns))


# # # Using bag of words feature extractor and Kneighbor classifier

# # In[20]:


# # RandomForestClassifier is used to build model
# from sklearn.neighbors import KNeighborsClassifier
# clf4 = MultiOutputClassifier(KNeighborsClassifier()).fit(X_train, Y_train)
# predicted = clf4.predict(X_test)


# # In[21]:


# print(generate_results(Y_test.iloc[:, :].values, predicted, Y_test.columns))



