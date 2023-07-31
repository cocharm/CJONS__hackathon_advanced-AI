import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# df = pd.read_csv('your_dataset.csv') 


df['review'] = df['review'].str.lower()  
df['review'] = df['review'].str.replace('[^\w\s]', '')  
df['review'] = df['review'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))  # 불용어 제거


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


feature_importances = clf.feature_importances_
feature_names = vectorizer.get_feature_names_out()
important_features = sorted(zip(feature_importances, feature_names), reverse=True)

print("Top important features:")
for importance, feature in important_features[:10]:  # 상위 10개 출력
    print(f"{feature}: {importance}")
