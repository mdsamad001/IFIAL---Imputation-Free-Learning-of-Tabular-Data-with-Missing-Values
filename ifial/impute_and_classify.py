from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

def impute_and_classify(train_dataset, test_dataset, y_train, y_test, val_dataset, y_val, seed=42):
    """
    Perform median imputation on numerical data, most frequent imputation on categorical and binary data,
    and classify using a Gradient Boosting Classifier.

    Args:
        train_dataset (DataFrame): Training dataset.
        test_dataset (DataFrame): Test dataset.
        y_train (Series): Training labels.
        y_test (Series): Test labels.
        val_dataset (DataFrame): Validation dataset.
        y_val (Series): Validation labels.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Training, validation, and test scores.
    """
    
    # Separate numerical, categorical, and binary columns
    numerical_cols = train_dataset.select_dtypes(include=['number']).columns
    non_numerical_cols = train_dataset.select_dtypes(exclude=['number']).columns  # All other columns

    # Define the column transformer for imputation and encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_cols),  # Median imputation for numerical columns
            ('non_num', Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), non_numerical_cols)  # Imputation and encoding for non-numerical columns
        ]
    )

    # Define the pipeline with Gradient Boosting Classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=seed))
    ])

    # Fit the model
    pipeline.fit(train_dataset, y_train)

    # Predict probabilities
    val_probs = pipeline.predict_proba(val_dataset)[:, 1]  # Probability estimates for the positive class
    test_probs = pipeline.predict_proba(test_dataset)[:, 1]  # Probability estimates for the positive class

    # Calculate AUC scores
    val_auc = roc_auc_score(y_val, val_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    return val_auc, test_auc
