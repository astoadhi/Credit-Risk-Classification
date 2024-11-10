import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import seaborn as sns

def load_data(fname):
    """Reads CSV data and performs basic preprocessing.

    This function reads data from a given CSV file, removes duplicates,
    and returns the data as a Pandas DataFrame.

    Args:
        fname (str): Path to the CSV file.

    Returns:
        data: DataFrame containing the preprocessed data.
    """

    if not isinstance(fname, str):
        raise TypeError("`fname` must be a string.")

    data = pd.read_csv(fname)
    print(f'Data shape  : {data.shape}')
    return data

def split_input_output(data, target_col):
    """
  Splits a dataset into input (X) and output (y).

  This function takes a DataFrame as input and the name of the target column as a string.
  The target column will be separated into the y variable, while the rest will be the X variable.

  Args:
    data (pd.DataFrame): DataFrame containing the complete dataset.
    target_col (str): Name of the column containing the target data.

  Returns:
    X (input) and y (output) both of them dataframe.
  """
    X = data.drop(columns=target_col)
    y = data[target_col]
    print(f'Original data shape : {data.shape}')
    print(f'X data shape        : {X.shape}')
    print(f'y data shape        : {y.shape}')
    return X, y

def split_train_test(X, y, test_size, random_state):
    """
  Splits a dataset into training and testing sets.

  This function uses stratified sampling to ensure equal class proportions in the training and testing sets.

  Args:
    X: Features (independent variable).
    y: Target (dependent variable).
    test_size: Proportion of data to be used as the testing set.
    random_state: Value to generate the same random sequence each time the function is called.

  Returns:
    X_train, X_test, y_train, and y_test : Dataframe that use X and y for train,valid,and test.
  """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)

    print(f"X_train shape   : {X_train.shape}")
    print(f"y_train shape   : {y_train.shape}")
    print(f"X_test shape    : {X_test.shape}")
    print(f"y_test shape    : {y_test.shape}")

    # Return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

def serialize_data(data,path):
   """
  Serializes data using joblib.

  Args:
    data: Data to be serialized (can be a list, array, or other Python object).
    path: Path to the file where the data will be saved.

  Returns:
    None: This function does not return anything, but saves the data to a file.
  """
   joblib.dump(data,path)

def deserialized_data(path):
    """
  Deserializes data using joblib.

  Args:
    path: Path to the file where the data is stored.

  Returns:
    The deserialized data.
  """

    data=joblib.load(path)
    return data