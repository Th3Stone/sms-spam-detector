import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib 


path = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
spam_db = pd.read_csv(path, sep='\t', header=None, names=['label', 'message'])
print(spam_db.head())
print("Total messages:", len(spam_db))

spam_db['label_num'] = spam_db['label'].map({'ham': 0, 'spam': 1})

y = spam_db['label_num']
X = spam_db['message']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

print("Original Text Example:", X_train.iloc[0])
print("\nVectorized Shape (Rows, Unique Words):", X_train_dtm.shape)


nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

y_pred = nb.predict(X_test_dtm)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion)
print("(Row 0 = Actual Ham, Row 1 = Actual Spam)")



joblib.dump(nb, 'spam_model.pkl')
joblib.dump(vect, 'spam_vectorizer.pkl')
