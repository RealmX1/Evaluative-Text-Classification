import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import confusion_matrix, f1_score
from src.utils.utils import compute_f1_scores_per_topic, rank_f1_scores


def predict(model: AutoModelForSequenceClassification, dataset: Dataset, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs inference on the test dataset and returns the true and predicted labels.
    Args:
        model (AutoModelForSequenceClassification): Trained model.
        dataset (Dataset): Test dataset.
        batch_size (int): Batch size for evaluation.
    Returns:
        Tuple[np.ndarray, np.ndarray]: (true_labels, predicted_labels)
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import time
    
    model.eval()
    true_labels = []
    pred_labels = []
    dataloader = DataLoader(dataset, batch_size=batch_size)
    device = next(model.parameters()).device
    
    # Calculate total batches for progress bar
    total_batches = len(dataloader)
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(dataloader, total=total_batches, desc="Making predictions")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
        
        # Update progress bar with estimated time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / (pbar.n + 1)
        estimated_time_remaining = avg_time_per_batch * (total_batches - (pbar.n + 1))
        pbar.set_postfix({'ETA': f'{estimated_time_remaining:.1f}s'})
    
    pbar.close()
    return np.array(true_labels), np.array(pred_labels)


def compute_confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray, label_map: Dict[int, str]) -> np.ndarray:
    """
    Computes the confusion matrix.
    Args:
        true_labels (np.ndarray): True labels.
        predicted_labels (np.ndarray): Predicted labels.
        label_map (Dict[int, str]): Mapping from label indices to label names.
    Returns:
        np.ndarray: Confusion matrix.
    """
    labels = list(label_map.keys())
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    return cm 