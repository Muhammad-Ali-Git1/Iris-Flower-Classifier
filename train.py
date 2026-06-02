import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load dataset
df = pd.read_csv("IRIS.csv")

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("iris_model.pkl", "wb"))

print("Model trained and saved successfully!")