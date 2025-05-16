import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def plot_confusion_matrix(cm, class_names, title, save_path=None, true_labels=None, pred_labels=None):
    """
    Plots a confusion matrix with numbers and percentages, styled like mockdata.py.
    Also computes and displays accuracy, precision, and recall if true_labels and pred_labels are provided.
    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (List[str]): List of class names.
        title (str): Plot title.
        save_path (Optional[str]): Path to save the plot.
        true_labels (Optional[List[int]]): True labels for metrics.
        pred_labels (Optional[List[int]]): Predicted labels for metrics.
    Returns:
        fig: The matplotlib figure object.
        metrics: dict of accuracy, precision, recall (if labels provided)
    """
    total = np.sum(cm)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm[i, j]/total:.1%})'
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()

    metrics = None
    
    if true_labels is not None and pred_labels is not None:
        acc = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, zero_division=0)
        
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        
        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        # Display metrics on the plot
        metric_text = f"Accuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}"
        # Place at bottom left inside axes (x=0.01, y=0.01 in axes fraction)
        ax.text(0.01, 0.01, metric_text, transform=ax.transAxes, fontsize=12, va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig, metrics

# Helper to format metrics for markdown
def metrics_to_markdown(metrics):
    if not metrics:
        return ''
    return (f"| Metric    | Value   |\n"
            f"|-----------|---------|\n"
            f"| Accuracy  | {metrics['accuracy']:.4f} |\n"
            f"| Precision | {metrics['precision']:.4f} |\n"
            f"| Recall    | {metrics['recall']:.4f} |\n"
            f"| F1        | {metrics['f1']:.4f} |\n")

def plot_ranked_f1_scores(ranked_f1, title, save_path=None, topic_support=None):
    """
    Plots a horizontal bar chart of ranked F1 scores per topic, and displays the number of entries (support) for each topic.
    Args:
        ranked_f1 (List[Tuple[str, float]]): List of (topic, F1 score) tuples.
        title (str): Plot title.
        save_path (Optional[str]): Path to save the plot.
        topic_support (Optional[Dict[str, int]]): Mapping from topic to number of entries.
    Returns:
        fig: The matplotlib figure object.
    """
    import pandas as pd
    topics, f1_scores = zip(*ranked_f1)
    if topic_support is not None:
        supports = [topic_support.get(topic, 0) for topic in topics]
        topic_labels = [f"{topic} (n={support})" for topic, support in zip(topics, supports)]
    else:
        supports = [None] * len(topics)
        topic_labels = list(topics)
    df = pd.DataFrame({'Topic': topics, 'F1': f1_scores, 'Support': supports, 'Label': topic_labels})
    df = df.sort_values('F1', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['Label'], df['F1'], color=sns.color_palette('viridis', len(df)))
    for i, (v, n) in enumerate(zip(df['F1'], df['Support'])):
        label = f'{v:.2f}'
        if n is not None:
            label += f'; (n={n})'
        ax.text(v + 0.01, i, label, va='center')
    ax.set_xlabel('F1 Score', fontsize=14)
    ax.set_ylabel('Topic (n = number of entries)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 1.0)
    avg_f1 = df['F1'].mean()
    ax.axvline(x=avg_f1, color='red', linestyle='--', alpha=0.7)
    ax.text(avg_f1 + 0.01, -0.5, f'Average: {avg_f1:.2f}', color='red')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig 

# --- F1 score per topic and ranking (moved from evaluator.py) ---

def compute_f1_scores_per_topic(true_labels: np.ndarray, predicted_labels: np.ndarray, test_df_with_topics, label_map: Dict[int, str], topic_column: str = 'topic') -> Dict[str, float]:
    """
    Computes F1 scores per topic.
    Args:
        true_labels (np.ndarray): True labels.
        predicted_labels (np.ndarray): Predicted labels.
        test_df_with_topics (pd.DataFrame): Test DataFrame with topic column.
        label_map (Dict[int, str]): Mapping from label indices to label names.
        topic_column (str): Name of the topic column.
    Returns:
        Dict[str, float]: Mapping from topic to F1 score.
    """
    # Ensure the test_df_with_topics is aligned with the predictions
    assert len(test_df_with_topics) == len(true_labels)
    topics = test_df_with_topics[topic_column].values
    topic_f1 = {}
    for topic in np.unique(topics):
        idx = np.where(topics == topic)[0]
        topic_true = true_labels[idx]
        topic_pred = predicted_labels[idx]
        _, _, f1, _ = precision_recall_fscore_support(topic_true, topic_pred)
        topic_f1[topic] = f1
    return topic_f1

def rank_f1_scores(f1_scores_per_topic: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Sorts the topic-wise F1 scores from highest to lowest.
    Args:
        f1_scores_per_topic (Dict[str, float]): F1 scores per topic.
    Returns:
        List[Tuple[str, float]]: Sorted list of (topic, F1 score).
    """
    return sorted(f1_scores_per_topic.items(), key=lambda x: x[1], reverse=True)

def compute_support(test_df_with_topics: pd.DataFrame, topic_column: str) -> Dict[str, int]:
    """
    Computes the support (number of entries) for each topic.
    Args:
        test_df_with_topics (pd.DataFrame): Test DataFrame with topic column.
    Returns:
    """
    return test_df_with_topics[topic_column].value_counts().to_dict()

# --- Custom metric implementations (naive binary only) ---
def accuracy_score(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    return np.mean(true_labels == pred_labels)

def precision_recall_fscore_support(true_labels, pred_labels, zero_division=0):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    # Assume binary, positive class is 0
    tp = np.sum((pred_labels == 0) & (true_labels == 0))
    fp = np.sum((pred_labels == 0) & (true_labels != 0))
    fn = np.sum((pred_labels != 0) & (true_labels == 0))
    print(f'tp: {tp}, fp: {fp}, fn: {fn}')
    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) > 0 else zero_division
    support = np.sum(true_labels == 0)
    return precision, recall, f1, support