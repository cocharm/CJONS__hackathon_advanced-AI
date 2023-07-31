import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import pandas as pd


reviews_df = pd.read_csv("product_reviews.csv") 
reviews = reviews_df['review'].tolist() 

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens


vectorizer = TfidfVectorizer(tokenizer=tokenize_text)


tfidf = vectorizer.fit_transform(reviews)


keywords = []
indices = tfidf.toarray().argsort()[:, ::-1]
features = vectorizer.get_feature_names_out()
for i, idx in enumerate(indices):
    keywords.append([features[i] for i in idx[:5]])  


positive_keywords = []
for keyword_list in keywords:
    for keyword in keyword_list:
        polarity = TextBlob(keyword).sentiment.polarity
        if polarity > 0:  
            positive_keywords.append(keyword)

print(positive_keywords)
