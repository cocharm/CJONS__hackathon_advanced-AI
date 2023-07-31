from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np


def tokenize(text):
    return text.split()


reviews = ["This is a fantastic product", "I don't like this product at all", 
           "This is the best product I've ever bought", "I hate this product, it's terrible"]


vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=stopwords.words('english'))
tfidf_matrix = vectorizer.fit_transform(reviews)

indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
top_features = [features[i] for i in indices[:10]] 


sentiment_analysis = pipeline("sentiment-analysis")


positive_keywords = []
for word in top_features:
    result = sentiment_analysis(word)[0]
    if result['label'] == 'POSITIVE':  
        positive_keywords.append(word)

print(positive_keywords)
