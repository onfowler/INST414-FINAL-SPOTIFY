import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading Spotify dataset...")
df = pd.read_csv('songs_normalize.csv')

# 1. Feature Engineering & Cleaning
# Many songs have multiple genres (e.g., "hip hop, pop"). 
# To make classification cleaner, we split the string and keep the primary genre.
df['primary_genre'] = df['genre'].apply(lambda x: x.split(',')[0].strip())

# We isolate the Top 4 most common genres to keep the model focused
top_genres = ['pop', 'hip hop', 'rock', 'Dance/Electronic']
df_filtered = df[df['primary_genre'].isin(top_genres)].copy()

# 2. Define Features (X) and Label (y)
features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df_filtered[features]
y = df_filtered['primary_genre']

# 3. Train / Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Classifier
print("Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = rf.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")

# 6. Isolate the "Wrong" Predictions for the Medium Post
df_test = df_filtered.loc[X_test.index].copy()
df_test['predicted_genre'] = y_pred
wrong = df_test[df_test['primary_genre'] != df_test['predicted_genre']]

print("\n--- 5 INCORRECT PREDICTIONS FOR ANALYSIS ---")
print(wrong[['artist', 'song', 'primary_genre', 'predicted_genre', 'energy', 'danceability']].head(5))