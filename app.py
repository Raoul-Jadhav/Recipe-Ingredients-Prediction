# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:45:52 2021

@author: RAHUL
"""

import pandas as pd
import numpy as np
import nltk
nltk.download ()

pd.set_option('display.max_colwidth', -1)
data = pd.read_csv('F:\My Data Science Projects\Recipe_Prediction\indian_food.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

# Cleaning the data
corpus = []
ps = PorterStemmer()

for i in range(0,data.shape[0]):
 # Cleaning special character from the message
    dish_name = re.sub(pattern='[^a-zA-Z]', repl=' ', string=data.name[i])
    
 # Converting the entire message into lower case  
    message = dish_name.lower()
    
 # Tokenizing the review by words
    words = dish_name.split()   
 
 # Removing the stop words
    words = [word for word in words if word not in set(stopwords.words('english'))]
    
 # Stemming the words
    words = [ps.stem(word) for word in words]

 # Joining the stemmed words
    dish_name = ' '.join(words)

 # Building a corpus of messages
    corpus.append(dish_name)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()


#print(vectorizer.get_feature_names())

from sklearn.metrics.pairwise import sigmoid_kernel
# Compute the sigmoid kernel
sig = sigmoid_kernel(X, X)

# Reverse mapping of indices and dish name
indices = pd.Series(data.index, index=data.name.str.lower()).drop_duplicates()

indices.head()
indices['gajar ka halwa']
list(enumerate(sig[indices['gajar ka halwa']]))

sorted(list(enumerate(sig[indices['gajar ka halwa']])), key=lambda x: x[1], reverse=True)

def recommend(title, sig=sig):
    try:
    # Get the index corresponding to original_title
      idx = indices[title.strip().lower()]

    # Get the pairwsie similarity scores 
      sig_scores = list(enumerate(sig[idx]))

    # Sort the ingredients 
      sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 1 most similar ingredients
      sig_scores = sig_scores[:1]

    # ingredients indices
      dish_indices = [i[0] for i in sig_scores]

    # Top 1 most similar ingredients
      return data['ingredients'].iloc[dish_indices],int(data['prep_time'].iloc[dish_indices].values),int(data['cook_time'].iloc[dish_indices].values),str(data['course'].iloc[dish_indices].values),str(data['state'].iloc[dish_indices].values)
    except:
        return("An error occurred..! Try with another Dish")
        
        
#print(recommend('Chicken Tikka masala'))        


#Flask

from flask import Flask,request,render_template
app = Flask(__name__)

@app.route('/home',methods=['GET'])
def home():
    return render_template('index1.html')

@app.route('/predict_ingredient',methods=['POST'])
def predict_ingredient():
       text = request.form['dish']
       output,prep_time,cook_time,course,state = recommend(text)
       return render_template('index1.html',r=output,pt=prep_time,ct=cook_time,c=str(course),s=state)
   
    
   
if __name__ == "__main__":
    app.run(debug=True)
    