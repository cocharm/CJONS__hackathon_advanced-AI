import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens


data = {
    'text': ["This product is amazing!",
             "I love this lotion, it works really well.",
             "Not satisfied with the results.",
             "Best purchase ever! Highly recommended."],
    'label': [1, 1, 0, 1]  
}

df = pd.DataFrame(data)


df['tokens'] = df['text'].apply(preprocess_text)


embedding_model = Word2Vec(df['tokens'], vector_size=100, window=5, min_count=1, workers=4)


def get_sentence_embedding(text):
    tokens = preprocess_text(text)
    embedding = np.mean([embedding_model.wv[word] for word in tokens], axis=0)
    return embedding


X = df['text'].apply(get_sentence_embedding).to_list()
y = df['label']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)


pos_review_indices = df[df['label'] == 1].index
pos_reviews = df.loc[pos_review_indices, 'text']
pos_review_embeddings = pos_reviews.apply(get_sentence_embedding).to_list()

pos_keywords = []
for i in range(len(embedding_model.wv.index_to_key)):
    keyword = embedding_model.wv.index_to_key[i]
    keyword_embedding = embedding_model.wv[keyword]
    similarity_scores = [np.dot(keyword_embedding, review_embedding) for review_embedding in pos_review_embeddings]
    avg_similarity = np.mean(similarity_scores)
    pos_keywords.append((keyword, avg_similarity))


pos_keywords = sorted(pos_keywords, key=lambda x: x[1], reverse=True)


print("Top 5 Positive Keywords:")
for keyword, similarity in pos_keywords[:5]:
    print(keyword, similarity)
