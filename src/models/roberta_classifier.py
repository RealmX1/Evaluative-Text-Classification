from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from typing import Any
import os

from peft import LoraConfig, get_peft_model, TaskType


def get_tokenizer(model_name: str = 'distilroberta-base') -> AutoTokenizer:
    """
    Loads the appropriate tokenizer for the specified DistilRoBERTa model.
    Args:
        model_name (str): Name of the model.
    Returns:
        AutoTokenizer: Tokenizer instance.
    """
    return AutoTokenizer.from_pretrained(model_name)


def build_direct_model(model_name: str = 'distilroberta-base', num_labels: int = 2) -> AutoModelForSequenceClassification:
    """
    Loads a pre-trained DistilRoBERTa base model and adds a classification head.
    Args:
        model_name (str): Name of the model.
        num_labels (int): Number of output labels.
    Returns:
        AutoModelForSequenceClassification: Model instance.
    """
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def build_lora_model(model_name: str = 'distilroberta-base', num_labels: int = 2, config: Any = None) -> AutoModelForSequenceClassification:
    """
    Loads a pre-trained DistilRoBERTa model and adds a LoRA adapter using peft.
    Args:
        model_name (str): Name of the model.
        num_labels (int): Number of output labels.
        lora_config (Any): LoRA configuration object.
    Returns:
        AutoModelForSequenceClassification: Model instance with LoRA adapter.
    """
    lora_config = LoraConfig(
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        r = config['lora_rank'],
        bias = config['lora_train_bias'],
        target_modules=["query", "key", "value"], # Which layer to apply LoRA, usually only apply on MultiHead Attention Layer
        task_type = 'SEQ_CLS'
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # save all modules of the original model to file
    with open(f'outputs/{model_name}_modules.txt', 'w') as f:
        f.write(str(model.modules))
    
    peft_model = get_peft_model(model, lora_config)
    parameter_count_str = print_number_of_trainable_model_parameters(peft_model)
    print(parameter_count_str)
    # save to file
    with open(f'outputs/{model_name}_LoRa_trainable_model_parameters.txt', 'w') as f:
        f.write(parameter_count_str)
    
    return peft_model

# Define a function that can print the trainable parameters 
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def train_model(model: AutoModelForSequenceClassification, train_dataset: Any, eval_dataset: Any, training_args: TrainingArguments) -> Trainer:
    """
    Sets up and runs the training process using Hugging Face Trainer.
    Args:
        model (AutoModelForSequenceClassification): Model to train.
        train_dataset (Any): Training dataset.
        eval_dataset (Any): Evaluation dataset.
        training_args (TrainingArguments): Training arguments.
    Returns:
        Trainer: Trainer instance after training.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    return trainer


def save_model(model: AutoModelForSequenceClassification, save_path: str):
    """
    Saves the trained model (or adapter) and tokenizer.
    Args:
        model (AutoModelForSequenceClassification): Model to save.
        save_path (str): Path to save the model.
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    if hasattr(model, 'config') and hasattr(model.config, 'tokenizer_class'):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model.config.tokenizer_class)
            tokenizer.save_pretrained(save_path)
        except Exception:
            pass