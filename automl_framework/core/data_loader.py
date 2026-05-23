import os
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

class DataLoader:
    """
     DataLoader handles downloading, loading, and splitting datasets for the regression task.
    Supports Kaggle API downloads, direct URL downloads (e.g., UCI datasets), and local file loading.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_from_kaggle(self, dataset_name: str) -> str:
        """
        Downloads a dataset from Kaggle using the Kaggle API.
        
        Args:
            dataset_name (str): Kaggle dataset identifier (e.g., 'sobhanmoosavi/us-accidents')
            
        Returns:
            str: Path to the downloaded and extracted dataset directory
        """
        print(f"[DataLoader] Attempting to download dataset '{dataset_name}' from Kaggle...")
        try:
            # Import kaggle here so it's not a hard dependency if not used
            import kaggle
            kaggle.api.authenticate()
            target_path = os.path.join(self.data_dir, dataset_name.replace("/", "_"))
            os.makedirs(target_path, exist_ok=True)
            
            kaggle.api.dataset_download_files(dataset_name, path=target_path, unzip=True)
            print(f"[DataLoader] Successfully downloaded and extracted to {target_path}")
            return target_path
        except ImportError:
            print("[DataLoader] ERROR: 'kaggle' package is not installed. Please install it using 'pip install kaggle'.")
            raise
        except Exception as e:
            print(f"[DataLoader] ERROR: Failed to download from Kaggle: {e}")
            print("[DataLoader] Make sure kaggle.json is configured in ~/.kaggle/ or equivalent location.")
            raise

    def download_from_url(self, url: str, filename: str) -> str:
        """
        Downloads a dataset from a direct URL (e.g., UCI repository).
        
        Args:
            url (str): Direct download URL
            filename (str): Name to save the file as in the data directory
            
        Returns:
            str: Path to the downloaded file
        """
        import urllib.request
        target_path = os.path.join(self.data_dir, filename)
        if os.path.exists(target_path):
            print(f"[DataLoader] File already exists at {target_path}. Skipping download.")
            return target_path
            
        print(f"[DataLoader] Downloading from {url} to {target_path}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print("[DataLoader] Download complete.")
            return target_path
        except Exception as e:
            print(f"[DataLoader] ERROR: Failed to download from URL: {e}")
            raise

    def load_dataset(self, filepath: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads a CSV/TSV/Parquet file and separates features from the target variable.
        
        Args:
            filepath (str): Path to the dataset file
            target_column (str): Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and Target (y)
        """
        print(f"[DataLoader] Loading data from {filepath}...")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.tsv') or filepath.endswith('.txt'):
            df = pd.read_csv(filepath, sep='\t')
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format for {filepath}")
            
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic preprocessing such as handling missing values and encoding categorical features.
        
        Args:
            X (pd.DataFrame): Raw features
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        # Placeholder for preprocessing pipeline.
        # Can include imputation, scaling, one-hot encoding, etc.
        X_processed = X.copy()
        
        # Handle numerical columns: simple median imputation
        num_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if X_processed[col].isnull().any():
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                
        # Handle categorical columns: simple frequency imputation & dummy encoding
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if X_processed[col].isnull().any():
                X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
        
        if len(cat_cols) > 0:
            X_processed = pd.get_dummies(X_processed, columns=cat_cols, drop_first=True)
            
        return X_processed

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: Optional[float] = None, random_state: int = 42) -> dict:
        """
        Splits data into train, validation (optional), and test sets.
        
        Args:
            X (pd.DataFrame): Preprocessed features
            y (pd.Series): Target variable
            test_size (float): Proportion of dataset to include in the test split
            val_size (Optional[float]): Proportion of dataset to include in the validation split (relative to the full dataset)
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing splits, e.g., {'X_train': ..., 'y_train': ..., 'X_test': ..., 'y_test': ...}
        """
        print(f"[DataLoader] Splitting data (test_size={test_size}, val_size={val_size})...")
        
        if val_size:
            # Adjust test_size to account for the first split
            first_split_size = test_size + val_size
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=first_split_size, random_state=random_state
            )
            
            # Split the temporary set into validation and test
            val_relative_size = val_size / first_split_size
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - val_relative_size), random_state=random_state
            )
            
            return {
                "X_train": X_train, "y_train": y_train,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_test, "y_test": y_test
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return {
                "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test
            }
