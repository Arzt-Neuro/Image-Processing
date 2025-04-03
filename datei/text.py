import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings

# Include the preprocess_df function definition here
def preprocess_df(df, numeric_columns=None, categorical_columns=None,
                  impute_numeric=True, impute_categorical=True,
                  numeric_strategy='median', categorical_strategy='most_frequent',
                  standardize=True, mandatory_columns=None, one_hot_codierung=False, encode_categorical=True,
                  get_processors=False, cat_var_threshold=5):
    """
    Perform imputation and standardization on a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to preprocess
    numeric_columns : list or None
        List of numeric column names to process. If None, will auto-detect numeric columns.
    categorical_columns : list or None
        List of categorical column names to process. If None, will auto-detect non-numeric columns.
    impute_numeric : bool
        Whether to impute missing values in numeric columns
    impute_categorical : bool
        Whether to impute missing values in categorical columns
    numeric_strategy : str
        Strategy for imputing numeric values. Options: 'mean', 'median', 'most_frequent', 'constant'
    categorical_strategy : str
        Strategy for imputing categorical values. Options: 'most_frequent', 'constant'
    standardize : bool
        Whether to standardize numeric columns (zero mean, unit variance)
    mandatory_columns : str or list
        Column(s) that should not contain NaN values. Rows with NaN in these columns will be dropped.
    encode_categorical : bool
        Whether to encode categorical columns as one-hot vectors

    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame
    dict
        Dictionary containing the fitted imputers and scalers for future use
    """
    # Create a copy if requested
    df = df.copy()

    # Drop rows with NaN in mandatory columns
    if mandatory_columns is not None:
        df = df.dropna(axis=0, subset=mandatory_columns)
        mandatory_df = df[mandatory_columns].copy()

    # Auto-detect column types if not specified
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    if categorical_columns is None:
        categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    # Initialize dictionary to store fitted preprocessors
    fitted_preprocessors = {
        'numeric_imputer': None,
        'categorical_imputer': None,
        'scaler': None,
        'encoder': None
    }

    # Process numeric columns
    if numeric_columns and impute_numeric:
        # Create and fit the imputer
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
        fitted_preprocessors['numeric_imputer'] = numeric_imputer

        # Standardize if requested
        if standardize:
            # Identify columns with more than 5 unique values
            columns_to_scale = []
            for col in numeric_columns:
                if df[col].nunique() > cat_var_threshold:
                    columns_to_scale.append(col)

            # Only standardize columns with more than 5 unique values
            if columns_to_scale:
                scaler = StandardScaler()
                df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
                fitted_preprocessors['scaler'] = scaler
                fitted_preprocessors['scaled_columns'] = columns_to_scale

    # Check for mixed data types in categorical columns
    if categorical_columns:
        for col in categorical_columns:
            # Convert to string to handle potential mixed types
            values = df[col].astype(str).dropna().tolist()
            has_numeric = False
            has_string = False
            for val in values:
                # Check if it's a numeric string (can be converted to float)
                try:
                    float(val)
                    has_numeric = True
                except ValueError:
                    # Not convertible to number, it's a non-numeric string
                    has_string = True
                # If we've found both types, no need to check further
                if has_numeric and has_string:
                    break
            if has_numeric and has_string:
                unique_values = df[col].unique()
                warnings.warn(f"WARNING: Column '{col}' contains mixed data types (numbers and strings)")
                warnings.warn(f"Unique values in '{col}': {unique_values}")

    # Process categorical columns
    if categorical_columns and impute_categorical:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
        fitted_preprocessors['categorical_imputer'] = categorical_imputer

    # One-hot encode categorical variables if requested
    if categorical_columns and one_hot_codierung:
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[categorical_columns])

        # Create new DataFrame with encoded columns
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(categorical_columns),
            index=df.index
        )

        # Drop original categorical columns and join encoded ones
        df = df.drop(columns=categorical_columns).join(encoded_df)
        fitted_preprocessors['encoder'] = encoder
    elif categorical_columns and encode_categorical:
        for col in categorical_columns:
            fitted_preprocessors['category_codes'] = {}
            if pd.api.types.is_string_dtype(df[col]):
                # Store original categories for future reference
                categories = df[col].astype('category').cat.categories
                # Convert to category codes
                df[col] = df[col].astype('category').cat.codes
                # Store mapping for decoding later if needed
                fitted_preprocessors['category_codes'][col] = {
                    'categories': categories,
                    'codes': pd.Series(range(len(categories)), index=categories)
                }
                # Ensure missing values are represented as NaN (cat.codes uses -1 for missing)
                df[col] = df[col].replace(-1, np.nan)

    # Check for object columns and raise error if found (when check_object_columns is True)
    data_deposit = df.values
    df = pd.DataFrame(data_deposit, columns=df.columns, index=df.index)
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        error_msg = f"Object data type found in columns: {object_columns}. " \
                    f"Please convert these columns to appropriate types before processing."
        raise TypeError(error_msg)
    if mandatory_columns is not None:
        df[mandatory_columns] = mandatory_df

    if get_processors:
        return df, fitted_preprocessors
    else:
        return df


def main():
    """
    Demonstrates the usage of the preprocess_df function with an example dataset.
    """
    # Create a sample DataFrame with both numeric and categorical data
    data = {
        'age': [25, 30, np.nan, 40, 35],
        'income': [50000, 60000, 55000, np.nan, 70000],
        'education': ['Bachelor', 'Master', np.nan, 'PhD', 'Bachelor'],
        'gender': ['Male', 'Female', 'Male', np.nan, 'Female'],
        'score': [85, 90, 78, 92, np.nan]
    }

    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)
    print("\n")

    # Example 1: Basic preprocessing with auto-detection
    print("Example 1: Basic preprocessing with auto-detection")
    processed_df = preprocess_df(df)
    print(processed_df)
    print("\n")

    # Example 2: Get preprocessors for future use
    print("Example 2: Get preprocessors for future use")
    processed_df, preprocessors = preprocess_df(df, get_processors=True)
    print("Processed DataFrame:")
    print(processed_df)
    print("\nPreprocessors:")
    for key, value in preprocessors.items():
        print(f"{key}: {type(value)}")
    print("\n")

    # Example 3: Custom processing with specified columns and one-hot encoding
    print("Example 3: Custom processing with specified columns and one-hot encoding")
    numeric_cols = ['age', 'income', 'score']
    categorical_cols = ['education', 'gender']

    processed_df = preprocess_df(
        df,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        numeric_strategy='mean',
        one_hot_codierung=True,
        mandatory_columns='age'
    )
    print(processed_df)
    print("\n")

    # Example 4: Process new data using saved preprocessors
    print("Example 4: Process new data using saved preprocessors")
    # Create new sample data
    new_data = {
        'age': [28, np.nan, 45],
        'income': [52000, 65000, np.nan],
        'education': ['Bachelor', 'PhD', 'Master'],
        'gender': ['Female', 'Male', np.nan],
        'score': [88, np.nan, 94]
    }

    new_df = pd.DataFrame(new_data)
    print("New DataFrame:")
    print(new_df)

    # Manually apply the preprocessors from Example 2
    print("\nProcessed new DataFrame using saved preprocessors:")
    # This would typically be done in a separate function that applies saved preprocessors
    # For demonstration, we'll just preprocess the new data directly
    new_processed_df = preprocess_df(
        new_df,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        numeric_strategy='mean'
    )
    print(new_processed_df)


if __name__ == "__main__":
    main()