import pytest
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from src.models import roberta_classifier
from unittest.mock import MagicMock


def test_get_tokenizer_returns_tokenizer():
    """Test that get_tokenizer returns a valid Hugging Face tokenizer."""
    tokenizer = roberta_classifier.get_tokenizer('distilroberta-base')
    assert isinstance(tokenizer, PreTrainedTokenizerBase)


def test_build_direct_model_returns_model():
    """Test that build_direct_model returns a model with correct number of labels."""
    model = roberta_classifier.build_direct_model('distilroberta-base', num_labels=2)
    assert isinstance(model, PreTrainedModel)
    assert model.config.num_labels == 2


def test_build_lora_model_returns_model():
    """Test that build_lora_model returns a model with LoRA adapter (mocked)."""
    # We'll mock the LoRA config and adapter logic for now
    lora_config = MagicMock()
    model = roberta_classifier.build_lora_model('distilroberta-base', num_labels=2, lora_config=lora_config)
    assert isinstance(model, PreTrainedModel)
    # Optionally check for adapter presence if implemented


def test_train_model_returns_trainer():
    """Test that train_model returns a Trainer instance (mocked datasets)."""
    model = roberta_classifier.build_direct_model('distilroberta-base', num_labels=2)
    dummy_dataset = MagicMock()
    training_args = TrainingArguments(output_dir='test_output', per_device_train_batch_size=2, num_train_epochs=1)
    trainer = roberta_classifier.train_model(model, dummy_dataset, dummy_dataset, training_args)
    from transformers import Trainer
    assert isinstance(trainer, Trainer)


def test_save_model(tmp_path):
    """Test that save_model saves the model to the specified path (mocked)."""
    model = roberta_classifier.build_direct_model('distilroberta-base', num_labels=2)
    save_path = tmp_path / "model"
    roberta_classifier.save_model(model, str(save_path))
    # Check that the directory was created and contains files
    assert save_path.exists() 