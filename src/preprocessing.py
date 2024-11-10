import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def ohe_transform(dataset, subset, prefix, ohe):
  """
  Transforms a categorical column in a DataFrame using a pre-trained OneHotEncoder.

  Args:
    dataset: The DataFrame to transform.
    subset: The name of the column to encode.
    prefix: The prefix for the encoded columns.
    ohe: A pre-trained OneHotEncoder.

  Returns:
    The transformed DataFrame.
  """

  # Validate input parameters
  if not isinstance(dataset, pd.DataFrame):
    raise RuntimeError("Fungsi ohe_transform: parameter dataset harus bertipe DataFrame!")
  if not isinstance(ohe, OneHotEncoder):
    raise RuntimeError("Fungsi ohe_transform: parameter ohe harus bertipe OneHotEncoder!")
  if not isinstance(prefix, str):
    raise RuntimeError("Fungsi ohe_transform: parameter prefix harus bertipe str!")
  if not isinstance(subset, str):
    raise RuntimeError("Fungsi ohe_transform: parameter subset harus bertipe str!")

  # Check if subset column exists in the DataFrame
  try:
    dataset.columns.get_loc(subset)
  except KeyError:
    raise RuntimeError("Fungsi ohe_transform: parameter subset string namun data tidak ditemukan dalam daftar kolom yang terdapat pada parameter dataset.")

  print("Fungsi ohe_transform: parameter telah divalidasi.")

  # Create a copy to avoid modifying the original DataFrame
  dataset = dataset.copy()

  # Print original column names
  print(f"Fungsi ohe_transform: daftar nama kolom sebelum dilakukan pengkodean adalah {list(dataset.columns)}.\n")

  # Create new column names for encoded columns
  col_names = [f"{prefix}_{col}" for col in ohe.categories_[0].tolist()]

  # Encode the specified column
  encoded = pd.DataFrame(ohe.transform(dataset[[subset]]).toarray(),
                         columns=col_names,
                         index=dataset.index)

  # Concatenate the encoded DataFrame with the original DataFrame
  dataset = pd.concat([dataset, encoded], axis=1)

  # Drop the original categorical column
  dataset.drop(columns=[subset], inplace=True)

  # Print new column names
  print(f"Fungsi ohe_transform: daftar nama kolom setelah dilakukan pengkodean adalah {list(dataset.columns)}.\n")

  return dataset