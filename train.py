import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("data/hospitality.csv")

# Select features + target
features = ["lead_time", "adr", "booking_changes",
            "previous_cancellations", "total_of_special_requests",
            "deposit_type", "customer_type"]

X = df[features]
y = df["is_canceled"]

# Encode categorical columns
le_deposit = LabelEncoder()
le_customer = LabelEncoder()
X["deposit_type"] = le_deposit.fit_transform(X["deposit_type"])
X["customer_type"] = le_customer.fit_transform(X["customer_type"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)


# Save model + encoders
os.makedirs("models", exist_ok=True)
joblib.dump((model, le_deposit, le_customer), "models/booking_model.pkl")

print(y_pred)
print(acc)
print("✅ Model trained and saved as models/booking_model.pkl")
