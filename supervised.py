import os
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

nltk.download('punkt')


data_dir = "빅데이터셋_디렉토리_경로"


def load_reviews(data_dir):
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
            for line in file:
                yield line.strip()


def tokenize(text):
    return word_tokenize(text)


vectorizer = TfidfVectorizer(tokenizer=tokenize)


aspects = ["카메라", "배터리", "디스플레이", "서비스"]
sentiments = ["긍정", "부정"]


def load_aspect_sentiments(data_dir):
    aspect_sentiments = []
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
            for line in file:
                aspect_sentiment_pairs = [pair.split(":") for pair in line.strip().split(",")]
                aspect_sentiments.append(aspect_sentiment_pairs)
    return aspect_sentiments


aspect_to_num = {aspect: i for i, aspect in enumerate(aspects)}
sentiment_to_num = {sentiment: i for i, sentiment in enumerate(sentiments)}


X = np.zeros((len(reviews), len(aspects)))
y = np.zeros(len(reviews))
for idx, review_aspects in enumerate(aspect_sentiments):
    for aspect, sentiment in review_aspects:
        X[idx, aspect_to_num[aspect]] = 1
        y[idx] = sentiment_to_num[sentiment]


classifier = MultinomialNB()
classifier.fit(X, y)


def extract_positive_keywords(reviews, vectorizer, classifier):
    
    tokenized_reviews = [tokenize(review) for review in reviews]
    review_vectors = vectorizer.transform([" ".join(tokens) for tokens in tokenized_reviews])

    
    sentiment_predictions = classifier.predict(review_vectors)

    
    positive_reviews_indices = np.where(sentiment_predictions == sentiment_to_num["긍정"])[0]

    
    positive_keywords = set()
    for idx in positive_reviews_indices:
        review_keywords = set(tokenized_reviews[idx])
        positive_keywords.update(review_keywords)

    return positive_keywords


positive_keywords = extract_positive_keywords(load_reviews(data_dir), vectorizer, classifier)
print("긍정적인 키워드:", positive_keywords)
