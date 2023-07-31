import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('vader_lexicon')


sia = SentimentIntensityAnalyzer()


reviews_df = pd.read_csv("product_reviews.csv") 


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews_df['review'])


sentiment_scores = [sia.polarity_scores(review) for review in reviews_df['review']]

positive_reviews = [review for review, score in zip(reviews_df['review'], sentiment_scores) if score['compound'] > 0]

print(positive_reviews)
