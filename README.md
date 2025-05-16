# Evaluative vs. Non-evaluative Text Classification

This project implements and evaluates a text classification pipeline to distinguish between "evaluative" and "non-evaluative" text snippets, based on a nuanced definition detailed in the associated research paper. It leverages LLM-labeled data and compares the performance of standard direct fine-tuning against Parameter-Efficient Fine-Tuning (PEFT) using LoRA on a pre-trained language model (DistilRoBERTa-base).

## Research Paper

This codebase accompanies the research paper titled: **"Beyond Sentiment: Classifying Evaluative vs. Non-evaluative Text using LLM-Labeled Data and Adapter Fine-tuning"** by Bomin Zhang. Refer to the paper for detailed definitions, methodology, results, and analysis.

## Project Structure

```
final_project/research/
├── data/
│   └── classification_results.csv
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── tests/
│   ├── ...
├── train_direct.py
├── train_lora.py
├── evaluate.py
├── requirements.txt
├── README.md
└── config.yaml
```
*(Note: The `data/raw/20_newsgroups/` and `data/processed/` directories are assumed based on standard practice and the paper's description of data processing. The actual raw 20 Newsgroups data is typically downloaded via libraries like scikit-learn or datasets, not stored directly in the repo.)*

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd final_project/research/
    ```
2.  **Install dependencies:**
    Ensure you have Python 3.7+ installed.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Acquire Data:**

    1. ** Pre Processing **
    
        run `prune_20_newsgroup.py` to prune the 20 Newsgroups dataset which would be saved to `data/20_newsgroup.csv`

    * **Option A (Use provided LLM-labeled data):** If `data/processed/classification_results.csv` is provided with the repository, you can skip the labeling step and proceed directly to training/evaluation.
    * **Option B (Generate LLM-labeled data):**
        * You need the raw 20 Newsgroups dataset. This can typically be loaded using libraries like `sklearn.datasets.fetch_20newsgroups`. Ensure your data loading script points to or downloads this data.
        * You need an OpenAI API key to use GPT-4.1 mini for labeling. Store your key securely (e.g., in an environment variable or a file not tracked by git). The script expects the key to be loaded. A common pattern is to use a `.env` file or read from `openai.key`. If using `openai.key`, create this file in the root directory of the project (`final_project/research/`) and paste your key inside. **Remember to keep `openai.key.template` and add `openai.key` to your `.gitignore`.**
        * Update relevant paths or parameters in `config.yaml` if necessary (e.g., path to raw data).
        * Run the labeling script:
            ```bash
            python scripts/run_llm_labeling.py
            ```
            This script will process the raw data and save the LLM-labeled output to `data/processed/classification_results.csv`. *Note: LLM labeling incurs costs based on API usage.*

4.  **Configuration:**
    Review `config.yaml` to adjust hyperparameters (learning rate, batch size, epochs), model names (should be `distilroberta-base` by default), data paths, or other settings if needed.

## Running the Pipeline

After setting up and preparing the data (`data/processed/classification_results.csv` must exist), you can run the training and evaluation scripts.

1.  **Direct Fine-tuning:**
    Train the DistilRoBERTa-base model using the standard fine-tuning approach.
    ```bash
    python scripts/train_direct.py
    ```
    *(Expected runtime: 5-10 minutes on an RTX 2080Ti as per the paper, adjust batch size in `config.yaml` if memory issues occur).*

2.  **LoRA Fine-tuning:**
    Train the DistilRoBERTa-base model using the LoRA PEFT method.
    ```bash
    python scripts/train_lora.py
    ```
    *(Expected runtime: Similar to direct fine-tuning).*

3.  **Evaluation:**
    Evaluate the trained model(s) and generate performance metrics and plots.
    ```bash
    python scripts/evaluate.py --model both # Evaluate both models (default)
    # Or
    python scripts/evaluate.py --model direct # Evaluate only the direct model
    # Or
    python scripts/evaluate.py --model lora   # Evaluate only the LoRA model
    ```
    Evaluation results, including confusion matrices and F1 scores per topic, will be saved to the `plots/` directory. A markdown file summarizing the graph information will also be generated there.

## Pipeline Overview

1.  **Data Acquisition & Labeling:** Download/load 20 Newsgroups data and label a subset using an LLM (GPT-4.1 mini) based on the specific evaluative/non-evaluative definition.
2.  **Data Preprocessing:** Clean texts, segment, encode labels, split data into training and test sets (80/20 split as per paper), and tokenize for the chosen model.
3.  **Model Training:** Fine-tune `DistilRoBERTa-base` on the LLM-labeled training data using two methods:
    * Standard End-to-End Fine-tuning.
    * LoRA Parameter-Efficient Fine-tuning.
4.  **Evaluation:** Evaluate the trained models on the held-out test set. Calculate overall metrics (Accuracy, Precision, Recall, F1) and analyze performance by topic.
5.  **Visualization:** Generate plots for confusion matrices and topic-wise performance.

## Data Format

The primary input file for training and evaluation scripts is expected to be `data/processed/classification_results.csv`. It should be a CSV file with at least the following columns:

* `topic`: The original topic/category of the text snippet (e.g., `sci.space`, `talk.religion.misc`). Used for per-topic analysis.
* `text`: The preprocessed text snippet.
* `classification`: The assigned label, expected to be one of the two classes defined in the paper (e.g., `evaluative`, `non-evaluative`).

The `run_llm_labeling.py` script is responsible for generating this file from raw data.

## Test-Driven Development

All core modules in the `src/` directory were developed using Test-Driven Development (TDD) principles. Unit tests are located in the `tests/` directory.

## Notes

* A GPU is highly recommended for efficient training and evaluation.
* Ensure sufficient disk space for the dataset and model checkpoints.
* The `config.yaml` file provides centralized configuration for the pipeline.

## License

This project is licensed under the MIT License

## Contact

Bomin Zhang (bominz2@umd.edu)