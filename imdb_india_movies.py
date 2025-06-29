import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

# Load dataset
df = pd.read_csv("imdb_india_movies.csv", encoding='latin1')

# Drop rows without target rating
df = df.dropna(subset=['Rating'])

# Handle 'Duration' safely
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Duration'] = df['Duration'].fillna(df['Duration'].median())

# Handle 'Votes' safely
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Votes'] = df['Votes'].fillna(df['Votes'].median())
df['LogVotes'] = np.log1p(df['Votes'])  # Log transform votes

# Combine Actor columns safely
df['Actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].fillna('').agg(lambda x: ','.join(filter(None, x)), axis=1)

# Convert genres to list
df['Genres'] = df['Genre'].fillna('').apply(lambda x: x.split('|'))

# Fill missing directors
df['Director'] = df['Director'].fillna('Unknown')

# === Feature Engineering ===

# Multi-label genre vectorization
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df['Genres'])

# Actor vectorization (top 100 only)
actor_vectorizer = CountVectorizer(max_features=100)
actor_matrix = actor_vectorizer.fit_transform(df['Actors'])

# Label encode director
le_director = LabelEncoder()
df['Director_encoded'] = le_director.fit_transform(df['Director'])
director_sparse = csr_matrix(df[['Director_encoded']].values)

# Normalize duration and votes
numeric_features = df[['Duration', 'LogVotes']]
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)
numeric_sparse = csr_matrix(numeric_scaled)

# Combine all features into final feature matrix
X = hstack([genre_matrix, actor_matrix, director_sparse, numeric_sparse])
y = df['Rating'].values

# Fix NaNs if any in feature matrix (convert to dense temporarily to detect)
if np.isnan(X.data).any():
    print("NaNs in feature matrix before fix:", np.isnan(X.data).sum())
    X.data = np.nan_to_num(X.data)
    print("NaNs in feature matrix after fix:", np.isnan(X.data).sum())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# Plot Actual vs Predicted Ratings
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.grid(True)
plt.tight_layout()
plt.show()