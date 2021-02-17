# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:45:52 2021

@author: RAHUL
"""

import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', -1)
data = pd.read_csv('indian_food.csv')

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')
X = vectorizer.fit_transform(data.name).toarray()


print(vectorizer.get_feature_names())

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

@app.route('/',methods=['GET'])
def home():
    return render_template('index1.html')

@app.route('/predict_ingredient',methods=['POST'])
def predict_ingredient():
       text = request.form['dish']
       output,prep_time,cook_time,course,state = recommend(text)
       return render_template('index1.html',r=output,pt=prep_time,ct=cook_time,c=str(course),s=state)
   
    
   
if __name__ == "__main__":
    app.run(debug=True)
    
