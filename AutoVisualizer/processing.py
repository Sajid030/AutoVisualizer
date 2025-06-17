# Module that will handle processing tasks of the user dataset
import numpy as np
import pandas as pd
import streamlit as st

# A quick cleanliness checker function
def check_dataset_cleanliness(df):
    issues_found = False

    # 1. Columns with null values
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        issues_found = True
        st.warning(f"⚠️ These columns contain missing (NaN) values: {null_cols}")

    # 2. Object columns that appear to be numeric (but aren't due to dirty values)
    misclassified_numeric = []
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna().astype(str)
        sample_size = min(100, len(non_null))
        if sample_size == 0:
            continue  # skip if column has no non-null values
        sample = non_null.sample(sample_size, random_state=1)
        numeric_like_ratio = sample.str.replace(",", "").str.replace(".", "", regex=False).str.isdigit().mean()
        if numeric_like_ratio > 0.6:
            misclassified_numeric.append(col)

    if misclassified_numeric:
        issues_found = True
        st.warning(
            f"⚠️ These columns are stored as `object` but mostly contain numeric values.\n"
            f"This may be due to the presence of invalid or non-numeric entries in a few rows: {misclassified_numeric}"
        )

    # 3. Checking duplicate records
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        issues_found = True
        st.warning(f"⚠️ Your dataset contains {num_duplicates} duplicated rows.")

    # 4. Checking Constant Columns (No Variation)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        issues_found = True
        st.warning(f"⚠️ These columns contain only a single unique value and may be useless for analysis: {constant_cols}")

    # 5. Suspiciously High Cardinality in Categorical Columns
    high_card_cols = [col for col in df.select_dtypes(include='object') if df[col].nunique() > 100]
    if high_card_cols:
        issues_found = True
        st.warning(f"⚠️ These object-type columns have unusually high unique values (possibly IDs or noisy data): {high_card_cols}")


    # Final message
    if not issues_found:
        st.success("✅ No major issues detected. Dataset looks clean!")


# Function that will identify the task of the dataset
def task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Determine the machine learning task type based on the target column.

    Args:
        df (pd.DataFrame): The input dataset.
        target_col (str or None): The target column name, or None for unsupervised tasks.

    Returns:
        str: One of the following task types:
            - "Classification" if the target is categorical.
            - "Clustering" if the target is not provided.
            - "Regression" if the target is numerical.
            - "Unknown" if the type cannot be recognized.
    """
    # Handle unsupervised case (no target)
    if target_col == "No Target":
        return "Clustering"
    
    target_series = df[target_col]
    dtype = target_series.dtype
    n_unique = target_series.nunique()
    
    # Check for classification
    if dtype == 'object' or dtype == 'bool' or (dtype == 'category'):
        return "Classification"
    
    # Check for binary/multi-class classification represented as integers or floats
    if dtype.kind in ['i', 'u', 'f']:  # Integer types
        if n_unique <= 10:  # Arbitrary threshold for classification
            return "Classification"
        else:
            return "Regression"
    
    # Numeric types default to regression
    if dtype.kind in ['i', 'u', 'f']:
        return "Regression"
    
    return "Unknown"

# Function that will identify if an object feature is truly categorical or not
def is_probably_categorical(series: pd.Series, threshold_unique: int = 50, threshold_ratio: float = 0.1) -> bool:
    """
    Determines whether a given pandas Series is likely to be a categorical feature.

    Args:
        series (pd.Series): The input data column to analyze.
        threshold_unique (int, optional (default=50)) : Maximum number of unique values for an object-type column to be considered categorical.
        threshold_ratio (float, optional (default=0.1)) : Maximum ratio of unique values to total entries for object-type column to be treated as categorical.
    
    Returns:
        bool: True if the series is likely categorical, False otherwise.
    """

    # Heuristic for object types (e.g., strings): avoid classifying high-cardinality fields as categorical
    if series.dtype == 'object':
        num_unique = series.nunique()
        unique_ratio = num_unique / len(series)
        
        if num_unique <= threshold_unique and unique_ratio <= threshold_ratio:
            return True   # categorical
        else:
            return False  # high-cardinality non-categorical (like names)
    
    # Explicit categorical or boolean data types are considered categorical
    elif pd.api.types.is_categorical_dtype(series):
        return True
    elif pd.api.types.is_bool_dtype(series):
        return True
    
    return False

# Function that will identify if an numerical feature is discrete or not
def is_discrete(series: pd.Series, max_unique: int = 20) -> bool:
    """
    Determine whether a numeric series should be considered discrete.

    Args:
        series (pd.Series): The input numeric data column to analyze.
        max_unique (int, optional): Maximum number of unique values allowed 
            to treat a column as discrete. Default is 20.

    Returns:
        bool: True if the series is likely discrete, False otherwise.
    """
    # Check if the series is of integer type
    if pd.api.types.is_integer_dtype(series):
        return series.nunique() <= max_unique
    
    if pd.api.types.is_float_dtype(series):
        # If all values are whole numbers AND unique count is low → treat as discrete
        if series.dropna().apply(float.is_integer).all():
            return series.nunique() <= max_unique
        
    return False

# Function that will identify if an numerical feature is continuous or not
def is_continuous(series: pd.Series, max_unique: int = 20) -> bool:
    """
    Determine whether a numeric series is continuous.

    Args:
        series (pd.Series): The input numeric data column to analyze.
        max_unique (int, optional): Threshold for unique values. If a float-type column
            contains only whole numbers and has fewer than this count, it is not considered continuous.
            Default is 20.

    Returns:
        bool: True if the series is likely continuous, False otherwise.
    """
    # Only float types are considered potentially continuous
    if pd.api.types.is_float_dtype(series):
        # If it's float but looks like discrete, then not continuous
        all_whole_numbers = series.dropna().apply(float.is_integer).all()
        if all_whole_numbers and series.nunique() <= max_unique:
            return False
        return True
    return False

# Function that will identify if an feature is date-time format and then extract the time-based components
def parse_datetime_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """
    Detects and parses datetime columns in a DataFrame, and extracts useful 
    date and/or time components into new columns.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        tuple:
            - pd.DataFrame: Updated DataFrame with extracted datetime components.
            - list: List of original columns identified as datetime.
            - list: List of newly extracted datetime-related feature names.
    """
    datetime_cols = []
    extracted_datetime = []
    today = pd.Timestamp.today()  # Just the date, no time

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == "object":
            try:
                converted = pd.to_datetime(df[col], errors="raise")
                df[col] = converted
                datetime_cols.append(col)
            except Exception:
                continue

    for col in datetime_cols:
        # Flags for what actually exists
        has_date = True
        has_time = True

        # Check if all dates are "today" → probably not originally present
        # if df[col].dt.normalize().nunique() == 1 and df[col].dt.normalize().iloc[0] == today:
        if (df[col].dt.year == today.year).all() or (df[col].dt.month == today.month).all() or (df[col].dt.day == today.day).all():
            has_date = False

        # Check if all times are 00:00:00 → probably not originally present
        if (df[col].dt.hour == 0).all() and (df[col].dt.minute == 0).all() and (df[col].dt.second == 0).all():
            has_time = False

        if has_date:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.day_name()
            extracted_datetime.extend([
                f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_weekday"
            ])
        else:
            df[f"{col}_year"] = np.nan
            df[f"{col}_month"] = np.nan
            df[f"{col}_day"] = np.nan
            df[f"{col}_weekday"] = np.nan

        if has_time:
            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_minute"] = df[col].dt.minute
            extracted_datetime.extend([
                f"{col}_hour", f"{col}_minute"
            ])
        else:
            df[f"{col}_hour"] = np.nan
            df[f"{col}_minute"] = np.nan
    
    # Remove any columns that are now entirely NaN
    df = df.dropna(axis=1, how='all')

    return df, datetime_cols, extracted_datetime
