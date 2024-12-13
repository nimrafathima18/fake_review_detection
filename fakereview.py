#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


# In[3]:


nltk.download('omw-1.4')


# In[4]:


data = pd.read_csv( r"C:\Users\nimra\OneDrive\Desktop\yelp.csv")
data.head()


# In[5]:


data.isnull().sum()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data['stars'].value_counts()


# In[9]:


data['length'] = data['text'].apply(len)
data.head()


# In[10]:


# COMPARING TEXT LENGTH TO STARS
graph = sns.FacetGrid(data=data,col='stars')
graph.map(plt.hist,'length',bins=50,color='blue')


# In[11]:


stval = data.groupby('stars').mean()
stval


# In[12]:


# FINDING THE CORRELATION BETWEEN THE VOTE COLUMNS
stval.corr()


# In[13]:


# CLASSIFICATION
data_classes = data[(data['stars']==1) | (data['stars']==3) | (data['stars']==5)]#select the rows where rating is 1,3,5
data_classes.head()#contains the rows with star value 1,3,5
print(data_classes.shape)#prints the number of rows and columns in the filtered data

# Seperate the dataset into X and Y for prediction
x = data_classes['text']#This selects the text column, likely representing customer reviews
y = data_classes['stars']#This selects the stars column
print(x.head())#Shows sample text data 
print(y.head())#Shows corresponding ratings 


# In[14]:


# CLEANING THE REVIEWS - REMOVAL OF STOPWORDS AND PUNCTUATION
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]#removing punctuation
    nopunc = ''.join(nopunc)# Rejoining Characters into Words
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]#Removing Stopwords


# In[15]:


data['text'][0], text_process(data['text'][0])


# In[16]:


data['text'].head().apply(text_process)


# In[17]:


data.shape


# In[18]:


data['text'] = data['text'].astype(str)


# In[19]:


def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])


# In[20]:


import nltk
nltk.download('punkt')


# In[21]:


preprocess(data['text'][4])


# In[22]:


data['text'][:2000] = data['text'][:2000].apply(preprocess)


# In[23]:


data['text'][2001:4000] = data['text'][2001:4000].apply(preprocess)


# In[24]:


data['text'][4001:6000] = data['text'][4001:6000].apply(preprocess)


# In[25]:


data['text'][6001:8000] = data['text'][6001:8000].apply(preprocess)


# In[26]:


data['text'][8001:10000] = data['text'][8001:10000].apply(preprocess)


# In[27]:


data['text'] = data['text'].str.lower()


# In[28]:


stemmer = PorterStemmer()#stemming Tool Initialization
def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])#Splits the text into individual words joins the list of stemmed words 
data['text'] = data['text'].apply(lambda x: stem_words(x))#Applying Stemming to the Dataset


# In[29]:


lemmatizer = WordNetLemmatizer()#Lemmatizer Initialization
def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
data["text"] = data["text"].apply(lambda text: lemmatize_words(text))


# In[30]:


data['text'].head()


# In[31]:


data.dropna(inplace=True)


# In[33]:


vocab = CountVectorizer(analyzer=text_process).fit(x)# Initialize CountVectorizer
print(len(vocab.vocabulary_))# Outputs the total number of unique words in the vocabulary.
r0 = x[0]#Selects and displays the first review from the text data x
print(r0)
vocab0 = vocab.transform([r0])#Converts the review r0 into a sparse vector representation
print(vocab0)
"""
    Now the words in the review number 78 have been converted into a vector.
    The data that we can see is the transformed words.
    If we now get the feature's name - we can get the word back!
"""
print("Getting the words back:")#Converts indices back into their corresponding words in the vocabulary.
print(vocab.get_feature_names_out()[19648])
print(vocab.get_feature_names_out()[10643])


# In[35]:


x = vocab.transform(x)
#Shape of the matrix:
print("Shape of the sparse matrix: ", x.shape)
#Non-zero occurences:
print("Non-Zero occurences: ",x.nnz)


# In[36]:


# DENSITY OF THE MATRIX
density = (x.nnz/(x.shape[0]*x.shape[1]))*100
print("Density of the matrix = ",density)


# In[37]:


# SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)


# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve


# In[40]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rmfr = RandomForestClassifier()
rmfr.fit(x_train,y_train)
predrmfr = rmfr.predict(x_test)
print("Confusion Matrix for Random Forest Classifier:")
print(confusion_matrix(y_test,predrmfr))
print("Score:",round(accuracy_score(y_test,predrmfr)*100,2))
print("Classification Report:",classification_report(y_test,predrmfr))


# In[41]:


# K Nearest Neighbour Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
predknn = knn.predict(x_test)
print("Confusion Matrix for K Neighbors Classifier:")
print(confusion_matrix(y_test,predknn))
print("Score: ",round(accuracy_score(y_test,predknn)*100,2))
print("Classification Report:")
print(classification_report(y_test,predknn))


# In[42]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression# Implements Logistic Regression
lr = LogisticRegression(solver='liblinear', random_state=0)
lr.fit(x_train,y_train)
predlr = lr.predict(x_test)#x_test: The test set of feature vectors,predlr: The predicted labels for the test data.
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test,predlr))#Outputs a confusion matrix to summarize the performance of the classification model.
print("Score: ",round(accuracy_score(y_test,predlr)*100,2))#Calculates and displays the accuracy of the model as a percentage.
print("Classification Report:")
print(classification_report(y_test,predlr))#Outputs a detailed report showing precision, recall, F1-score, and support


# In[ ]:




