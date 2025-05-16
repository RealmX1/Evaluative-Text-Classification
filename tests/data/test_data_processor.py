import pytest
import pandas as pd
import os
from src.data import data_processor
from transformers import AutoTokenizer

@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        'classification': ['pos', 'neg', 'pos', 'neg'],
        'topic': ['t1', 't2', 't1', 't2'],
        'text': ['foo', 'bar', 'baz', 'qux'],
        'analysis': ['x', 'y', 'z', 'w']
    })

def test_load_data_returns_dataframe(tmp_path):
    """Test that load_data returns a pandas DataFrame."""
    # Create a dummy CSV
    dummy = pd.DataFrame({
        'classification': ['a', 'b'],
        'topic': ['t1', 't2'],
        'text': ['foo', 'bar'],
        'analysis': ['x', 'y']
    })
    csv_path = tmp_path / "dummy.csv"
    dummy.to_csv(csv_path, index=False)
    df = data_processor.load_data(str(csv_path))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['classification', 'topic', 'text', 'analysis']
    assert len(df) == 2

def test_add_index_adds_column(dummy_df):
    """Test that add_index adds a unique integer index column."""
    df = data_processor.add_index(dummy_df)
    assert 'original_index' in df.columns
    assert df['original_index'].is_unique
    assert pd.api.types.is_integer_dtype(df['original_index'])
    assert (df['original_index'] == range(len(df))).all()

def test_encode_labels_creates_mapping_and_column(dummy_df):
    """Test that encode_labels creates a new integer column and returns the mapping dicts."""
    df, label2id, id2label = data_processor.encode_labels(dummy_df, 'classification')
    assert 'label_id' in df.columns
    assert set(df['label_id']) == set(label2id.values())
    assert all(label2id[label] == id2label_inv for label, id2label_inv in label2id.items())
    assert set(label2id.keys()) == set(dummy_df['classification'].unique())
    assert set(id2label.keys()) == set(label2id.values())

def test_split_data_correct_sizes(dummy_df):
    """Test that split_data returns train/test sets with correct sizes and stratification."""
    df, label2id, id2label = data_processor.encode_labels(dummy_df, 'classification')
    train_df, test_df = data_processor.split_data(df, 'text', 'label_id', test_size=0.5, random_state=42)
    assert len(train_df) + len(test_df) == len(df)
    # Check stratification: label distribution should be similar
    train_dist = train_df['label_id'].value_counts(normalize=True)
    test_dist = test_df['label_id'].value_counts(normalize=True)
    for label in train_dist.index:
        assert abs(train_dist[label] - test_dist[label]) < 0.5  # loose check

def test_tokenize_data_returns_hf_dataset(dummy_df):
    """Test that tokenize_data returns a Hugging Face Dataset object with correct fields."""
    df, label2id, id2label = data_processor.encode_labels(dummy_df, 'classification')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    dataset = data_processor.tokenize_data(df, 'text', 'label_id', tokenizer, max_length=8)
    assert 'input_ids' in dataset.features
    assert 'attention_mask' in dataset.features
    assert 'labels' in dataset.features
    assert len(dataset) == len(df) 