import numpy as np
import os
import tempfile
import pytest
from src.utils import utils
import matplotlib.figure

def test_plot_confusion_matrix_creates_figure_and_file():
    cm = np.array([[10, 2], [3, 15]])
    class_names = ['A', 'B']
    title = 'Test Confusion Matrix'
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'cm.png')
        fig = utils.plot_confusion_matrix(cm, class_names, title, save_path=save_path)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert os.path.exists(save_path)


def test_plot_ranked_f1_scores_creates_figure_and_file():
    ranked_f1 = [('topic1', 0.8), ('topic2', 0.6), ('topic3', 0.9)]
    title = 'Ranked F1'
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'f1.png')
        fig = utils.plot_ranked_f1_scores(ranked_f1, title, save_path=save_path)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert os.path.exists(save_path) 