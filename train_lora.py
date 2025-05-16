# train_lora.py
"""
Script to train RoBERTa with LoRA adapter fine-tuning.
"""

import os
import yaml
from src.data import data_processor
from src.models import roberta_classifier
from transformers import TrainingArguments
import wandb

from adapters import LoRAConfig

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)


    print('Loading data...')
    df = data_processor.load_data(config['raw_data_path'])
    print('Adding index...')
    df = data_processor.add_index(df)
    print('Encoding labels...')
    df, label2id, id2label = data_processor.encode_labels(df, 'classification')
    print('Splitting data...')
    train_df, test_df = data_processor.split_data(df, 'text', 'label_id', test_size=config['test_size'], random_state=config['random_state'])
    print('Loading tokenizer...')
    tokenizer = roberta_classifier.get_tokenizer(config['model_name'])
    print('Tokenizing train set...')
    train_dataset = data_processor.tokenize_data(train_df, 'text', 'label_id', tokenizer, max_length=config['max_length'])
    print('Tokenizing test set...')
    test_dataset = data_processor.tokenize_data(test_df, 'text', 'label_id', tokenizer, max_length=config['max_length'])
    print('Building LoRA model...')
    model = roberta_classifier.build_lora_model(config['model_name'], num_labels=config['num_labels'], config=config)
    print('Setting up training arguments...')
    wandb.init(project=config.get('wandb_project', 'roberta-lora'), config=config)
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['lora_epochs'],
        per_device_train_batch_size=config['lora_batch_size'],
        per_device_eval_batch_size=config['lora_batch_size'],
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=config['lora_learning_rate'],
        logging_dir=os.path.join(config['output_dir'], 'logs'),
        logging_steps=10,
        report_to='wandb',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )
    print('Starting LoRA training... (ETA will be shown by Hugging Face Trainer logs)')
    trainer = roberta_classifier.train_model(model, train_dataset, test_dataset, training_args)
    trainer.train()
    print('Saving LoRA adapter model...')
    roberta_classifier.save_model(model, os.path.join(config['output_dir'], 'lora_model'))
    print('LoRA training complete.')

if __name__ == "__main__":
    main() 