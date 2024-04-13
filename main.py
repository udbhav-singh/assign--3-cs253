import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB

def convert_to_numeric(amount):
    if 'Crore' in amount:
        return float(amount.replace(' Crore+', '')) * 10000  # 1 Crore = 10000 lacs
    elif 'Lac' in amount:
        return float(amount.replace(' Lac+', ''))   # Value is already in lacs
    elif 'Thou' in amount:
        return float(amount.replace(' Thou+', '')) / 100  # Convert thousands to lacs
    elif 'Hund' in amount:
        return float(amount.replace(' Hund+', '')) / 10000   # Convert hundreds to lacs
    else:
        return float(amount)

# Load the data
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Convert Total Assets and Liabilities to numerical values and apply Min-Max scaling
scaler = MinMaxScaler()
df['Total Assets'] = scaler.fit_transform(df['Total Assets'].apply(convert_to_numeric).values.reshape(-1, 1))
df['Liabilities'] = scaler.fit_transform(df['Liabilities'].apply(convert_to_numeric).values.reshape(-1, 1))

df_test['Total Assets'] = scaler.transform(df_test['Total Assets'].apply(convert_to_numeric).values.reshape(-1, 1))
df_test['Liabilities'] = scaler.transform(df_test['Liabilities'].apply(convert_to_numeric).values.reshape(-1, 1))

# Define features and target
features = ['Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state', 'Candidate']
target = 'Education'

# Create the Candidate field
df['Candidate'] = df['Candidate'].apply(lambda x: 1 if x.startswith('Dr.') or x.startswith('Adv.') else 0)
df_test['Candidate'] = df_test['Candidate'].apply(lambda x: 1 if x.startswith('Dr.') or x.startswith('Adv.') else 0)

# Preprocessing for numerical features
numeric_features = ['Criminal Case', 'Total Assets', 'Liabilities']
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

# Preprocessing for categorical features
categorical_features = ['Party', 'state']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model (Bernoulli Naive Bayes)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', BernoulliNB())
])

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'classifier__alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model from grid search
best_model = grid_search.best_estimator_

# Train best model on full training data
best_model.fit(df[features], df[target])

# Make predictions on the test data
predictions_test = best_model.predict(df_test[features])

# Write the predictions to a CSV file (for test data)
submission_df = pd.DataFrame({'ID': df_test['ID'], 'Education': predictions_test})
submission_df.to_csv('submission.csv', index=False)
