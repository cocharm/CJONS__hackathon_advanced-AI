import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('stopwords')


reviews = [...]  



def clean_text(text):
    
    text = text.lower()
    
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in stemmed_tokens if token not in stop_words]
    
    text = ' '.join(filtered_tokens)
    return text


cleaned_reviews = [clean_text(review) for review in reviews]


vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(cleaned_reviews)
keywords = vectorizer.get_feature_names_out()


labels = [1]*len(positive_labels) + [0]*len(negative_labels)


tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(reviews)


X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


model = Sequential()
model.add(Embedding(1000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)


score, acc = model.evaluate(X_test, Y_test, batch_size=32)
print("Test score: ", score)
print("Test accuracy: ", acc)


positive_keywords = []
for keyword in keywords:
    sequence = tokenizer.texts_to_sequences([keyword])
    pad_sequence = pad_sequences(sequence, maxlen=128)  # LSTM의 input_length에 따라 조정해야 합니다.
    prediction = model.predict(pad_sequence)
    if prediction > 0.5:
        positive_keywords.append(keyword)
print(positive_keywords)
