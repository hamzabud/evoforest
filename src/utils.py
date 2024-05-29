import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer



def prepare_data(train, target_col, drop_columns=None, seed=random.randint(0, 1000)):
    X = train.drop(columns=target_col)
    X = train.drop(columns=drop_columns) if drop_columns else X
    y = train[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.7, shuffle=True, random_state=seed)
    
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=True, random_state=seed)

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns


    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    
    for feature in categorical_features:
        print(f"{feature}: {train[feature].nunique()} unique categories")

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'y_val': y_val,
        'y_train': y_train,
        'y_test': y_test,
        'stats': calculate_feature_statistics(X_train, numerical_features.tolist(), categorical_features.tolist())
    }


def calculate_feature_statistics(data, numerical_features, categorical_features):
    stats = {}
    for i, feature_name in enumerate(numerical_features + categorical_features):
        stats[feature_name] = {
            'mean': data[:, i].mean(),
            'std_dev': data[:, i].std(),
            'min': data[:, i].min(),
            'max': data[:, i].max(),
        }
    return stats
