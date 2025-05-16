import pandas as pd
from typing import Tuple, Dict
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from the specified CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(filepath)


def add_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a unique integer index column to the DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with an added 'original_index' column.
    """
    df = df.copy()
    df['original_index'] = range(len(df))
    return df


def encode_labels(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Converts text labels into integers. Returns the modified DataFrame, a label-to-int mapping, and an int-to-label mapping.
    Args:
        df (pd.DataFrame): Input DataFrame.
        label_column (str): Name of the label column.
    Returns:
        Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]
    """
    df = df.copy()
    unique_labels = sorted(df[label_column].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    df['label_id'] = df[label_column].map(label2id)
    return df, label2id, id2label


def split_data(df: pd.DataFrame, text_column: str, label_column: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and testing sets.
    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the text column.
        label_column (str): Name of the label column.
        test_size (float): Proportion of test set.
        random_state (int): Random seed.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column] if label_column in df.columns else None
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_data(dataframe: pd.DataFrame, text_column: str, label_column: str, tokenizer: AutoTokenizer, max_length: int = 128) -> Dataset:
    """
    Tokenizes the text column and prepares a Hugging Face Dataset object.
    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the text column.
        label_column (str): Name of the label column.
        tokenizer (AutoTokenizer): Tokenizer to use.
        max_length (int): Max token length.
    Returns:
        Dataset: Hugging Face Dataset object.
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch[text_column],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column(label_column, 'labels')
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset 