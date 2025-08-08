import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Load dataset from local file
df = pd.read_csv("winequality-red.csv", sep=';')  # Local file path

# Create labels
df['label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
df.drop('quality', axis=1, inplace=True)

X = df.drop('label', axis=1)
y = df['label']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model, scaler, PCA
pickle.dump(model, open("wine_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(pca, open("pca.pkl", "wb"))

print("✅ Model, Scaler, and PCA saved!")
