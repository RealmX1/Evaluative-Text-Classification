# evaluate.py
"""
Script to evaluate both direct and LoRA fine-tuned RoBERTa models.
"""

# Imports (to be implemented)
# from src.data.data_processor import ...
# from src.models.roberta_classifier import ...
# from src.evaluation.evaluator import ...
# from src.utils.utils import ...

import os
import yaml
import pandas as pd
from src.data import data_processor
from src.models import roberta_classifier
from src.evaluation import evaluator
from src.utils import utils

from adapters import LoRAConfig
import argparse
import torch

# Helper function to write to markdown file
def write_to_md(section_title, content, md_path):
    with open(md_path, 'a') as f:
        f.write(f'## {section_title}\n\n')
        f.write(content)
        f.write('\n\n')

def evaluate_direct_model(config, test_dataset, test_df, id2label):
    print('Loading direct model...')
    direct_model_path = os.path.join(config['output_dir'], 'direct_model')
    direct_model = roberta_classifier.AutoModelForSequenceClassification.from_pretrained(direct_model_path)
    print('Predicting with direct model...')
    true_labels, direct_preds = evaluator.predict(direct_model, test_dataset, batch_size=config['batch_size'])
    print('Computing confusion matrix for direct model...')
    cm_direct = evaluator.compute_confusion_matrix(true_labels, direct_preds, id2label)
    print('Plotting confusion matrix for direct model...')
    os.makedirs(config['plots_dir'], exist_ok=True)
    cm_path = os.path.join(config['plots_dir'], 'confusion_matrix_direct.png')
    _, metrics_direct = utils.plot_confusion_matrix(
        cm_direct, list(id2label.values()), 'Direct Model Confusion Matrix', save_path=cm_path,
        true_labels=true_labels, pred_labels=direct_preds
    )
    # Write confusion matrix to markdown
    md_path = os.path.join(config['plots_dir'], 'evaluation_report.md')
    cm_md = pd.DataFrame(cm_direct, index=list(id2label.values()), columns=list(id2label.values())).to_markdown()
    write_to_md('Direct Model Confusion Matrix', cm_md, md_path)
    # Write metrics to markdown
    metrics_md = utils.metrics_to_markdown(metrics_direct)
    write_to_md('Direct Model Metrics', metrics_md, md_path)

    print('Computing F1 scores per topic for direct model...')
    f1_direct = utils.compute_f1_scores_per_topic(true_labels, direct_preds, test_df, id2label, topic_column='topic')
    ranked_f1_direct = utils.rank_f1_scores(f1_direct)
    topic_support = utils.compute_support(test_df, 'topic')
    print('Ranked F1 scores per topic (Direct Model):')
    f1_md = '| Topic | F1 Score |\n|-------|----------|\n'
    for topic, score in ranked_f1_direct:
        print(f'{topic}: {score:.4f}')
        f1_md += f'| {topic} | {score:.4f} |\n'
    ranked_f1_path = os.path.join(config['plots_dir'], 'ranked_f1_direct.png')
    utils.plot_ranked_f1_scores(ranked_f1_direct, 'Direct Model Ranked F1 Score by Topic', save_path=ranked_f1_path, topic_support=topic_support)
    write_to_md('Direct Model Ranked F1 Score by Topic', f1_md, md_path)

    # remove the model from memory
    del direct_model
    torch.cuda.empty_cache()

def evaluate_lora_model(config, test_dataset, test_df, id2label, true_labels=None):
    print('Loading LoRA model...')
    lora_model_path = os.path.join(config['output_dir'], 'lora_model')
    lora_model = roberta_classifier.AutoModelForSequenceClassification.from_pretrained(lora_model_path)
    print('Predicting with LoRA model...')
    if true_labels is None:
        true_labels, lora_preds = evaluator.predict(lora_model, test_dataset, batch_size=config['lora_batch_size'])
    else:
        _, lora_preds = evaluator.predict(lora_model, test_dataset, batch_size=config['lora_batch_size'])
    print('Computing confusion matrix for LoRA model...')
    cm_lora = evaluator.compute_confusion_matrix(true_labels, lora_preds, id2label)
    print('Plotting confusion matrix for LoRA model...')
    os.makedirs(config['plots_dir'], exist_ok=True)
    cm_path = os.path.join(config['plots_dir'], 'confusion_matrix_lora.png')
    _, metrics_lora = utils.plot_confusion_matrix(
        cm_lora, list(id2label.values()), 'LoRA Model Confusion Matrix', save_path=cm_path,
        true_labels=true_labels, pred_labels=lora_preds
    )
    # Write confusion matrix to markdown
    md_path = os.path.join(config['plots_dir'], 'evaluation_report.md')
    cm_md = pd.DataFrame(cm_lora, index=list(id2label.values()), columns=list(id2label.values())).to_markdown()
    write_to_md('LoRA Model Confusion Matrix', cm_md, md_path)
    # Write metrics to markdown
    metrics_md = utils.metrics_to_markdown(metrics_lora)
    write_to_md('LoRA Model Metrics', metrics_md, md_path)

    print('Computing F1 scores per topic for LoRA model...')
    f1_lora = utils.compute_f1_scores_per_topic(true_labels, lora_preds, test_df, id2label, topic_column='topic')
    ranked_f1_lora = utils.rank_f1_scores(f1_lora)
    topic_support = utils.compute_support(test_df, 'topic')
    print('Ranked F1 scores per topic (LoRA Model):')
    f1_md = '| Topic | F1 Score |\n|-------|----------|\n'
    for topic, score in ranked_f1_lora:
        print(f'{topic}: {score:.4f}')
        f1_md += f'| {topic} | {score:.4f} |\n'
    ranked_f1_path = os.path.join(config['plots_dir'], 'ranked_f1_lora.png')
    utils.plot_ranked_f1_scores(ranked_f1_lora, 'LoRA Model Ranked F1 Score by Topic', save_path=ranked_f1_path, topic_support=topic_support)
    write_to_md('LoRA Model Ranked F1 Score by Topic', f1_md, md_path)

    # remove the model from memory
    del lora_model
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Evaluate models (direct and/or LoRA)')
    parser.add_argument('--model', choices=['direct', 'lora', 'both'], default='both', help='Which model(s) to evaluate')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # clear md file
    md_path = os.path.join(config['plots_dir'], 'evaluation_report.md')
    with open(md_path, 'w') as f:
        f.write('')

    # Load and preprocess data
    print('Loading data...')
    df = data_processor.load_data(config['raw_data_path'])
    df = data_processor.add_index(df)
    df, label2id, id2label = data_processor.encode_labels(df, 'classification')
    print('Splitting data...')
    _, test_df = data_processor.split_data(df, 'text', 'label_id', test_size=config['test_size'], random_state=config['random_state'])

    # Get tokenizer
    print('Loading tokenizer...')
    tokenizer = roberta_classifier.get_tokenizer(config['model_name'])
    print('Tokenizing test set...')
    test_dataset = data_processor.tokenize_data(test_df, 'text', 'label_id', tokenizer, max_length=config['max_length'])

    if args.model == 'direct':
        evaluate_direct_model(config, test_dataset, test_df, id2label)
    elif args.model == 'lora':
        # For LoRA, we need true_labels for confusion matrix and F1
        true_labels, _ = evaluator.predict(
            roberta_classifier.build_direct_model(config['model_name'], num_labels=config['num_labels']),
            test_dataset
        ) if False else (None, None)  # Placeholder: if you want to use direct model's true_labels
        evaluate_lora_model(config, test_dataset, test_df, id2label, true_labels=true_labels)
    else:
        # Evaluate both
        print('Evaluating direct model...')
        true_labels, _ = evaluator.predict(
            roberta_classifier.build_direct_model(config['model_name'], num_labels=config['num_labels']),
            test_dataset
        ) if False else (None, None)  # Placeholder: if you want to use direct model's true_labels
        evaluate_direct_model(config, test_dataset, test_df, id2label)
        print('Evaluating LoRA model...')
        evaluate_lora_model(config, test_dataset, test_df, id2label, true_labels=true_labels)

if __name__ == "__main__":
    main() 