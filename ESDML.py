import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv(r"C:\Users\manas\Downloads\spam.csv", encoding='latin-1')

print("Dataset Preview:\n", data.head())

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

print("\nCleaned Dataset:\n", data.head())

X = data['message']
y = data['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Spam Detection')
plt.show()

custom_message = ["Congratulations! You have won $1000. Click here to claim your prize."]
custom_message_vec = vectorizer.transform(custom_message)
prediction = model.predict(custom_message_vec)
print("\nCustom Email Prediction:")
print("Spam" if prediction[0] == 1 else "Ham")
