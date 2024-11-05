import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Generating labels for classification (Example: age classification into groups)
df['Age_Group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60], labels=[0, 1, 2])

# Splitting dataset into features and target
X = df[['Height', 'Weight']]  # Features
y = df['Age_Group']  # Target (Age groups)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Making predictions
y_pred = knn.predict(X_test_scaled)

# Checking accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy
