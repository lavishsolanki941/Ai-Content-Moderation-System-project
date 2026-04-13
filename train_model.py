import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Data Collection (Mocking the Jigsaw Dataset load)
print("Loading Jigsaw Toxic Comment Dataset...")
# df = pd.read_csv('train.csv') # In a real scenario, you load the CSV here

# Mock data for demonstration of partial implementation
data = {'comment_text': ['You are an idiot', 'Have a great day', 'I will kill you', 'Nice picture'],
        'toxic': [1, 0, 1, 0],
        'threat': [0, 0, 1, 0]}
df = pd.DataFrame(data)

# 2. Data Preprocessing & Feature Extraction
print("Applying TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['comment_text'])
y_toxic = df['toxic']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_toxic, test_size=0.2, random_state=42)

# 3. Model Training (Logistic Regression)
print("Training Logistic Regression Model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Training Complete. Initial Accuracy: {accuracy * 100}%")