# Recipe-Ingredients-Prediction
# Overview
<h3><p>Hello guys,This is Indian Recipe Ingredient Prediction Project build using nltk library.</p></h3>
<h3><p>Here I have taken a dataset from Kaggle. Dataset contains recipe name,ingredients,preparation time and cooking time, thier belonging state names etc</p></h3>
<h3><p>I have use sigmoid kernel,TfidfVectorizer and other nltk libraries to predict recipe ingredients based on recipe name</p></h3>
<h3><p>First all recipe name are converted to vectors using TfidfVectorizer and vectors are pass to sigmoid kernel activation</p></h3>
<h3><p>Reverse mapping of indices and dish name are created. and preditions are done using pairwsie similarity scores.</p></h3>
<h3>Flask Web Framework is used for deployment and application is hosted on Heroku server.</h3>
<img src="/Flask.PNG" alt="" width="500" height="300">

# Screens
<img src="/Screen1.png" alt="" width="1200" height="450">
<img src="/Screen2.png" alt="" width="1200" height="450">
demo- https://recipeingredientsprediction.herokuapp.com/
