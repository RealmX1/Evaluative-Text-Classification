# Evaluative text Classification Research

This project demonstrates fine-tuning a distilRoBERTa-base model for text classification using both direct fine-tuning and LoRA adapter-based fine-tuning. The pipeline includes data processing, model training, evaluation, and visualization.

## Project Structure

```
final_project/research/
├── data/
│   └── raw/
│       └── classification_results.csv
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── tests/
├── scripts/
│   ├── train_direct.py
│   ├── train_lora.py
│   └── evaluate.py
├── requirements.txt
├── README.md
└── config.yaml
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (optional) store your openai key in `openai.key.template`, remove the `.template` suffix.
3. (optional) run `python LLM_labeling.py` to label your data with LLM.
4. (optional) Replace the data at `data/classification_results.csv` with your own data with columns: `topic`, `text`, `classification`.
5. (optional) Update the model name at `config.yaml`.
6. Run the following command to directly fine-tune the model: (should only take 5~10 minutes with batch size of 200, adjust batch size in `config.yaml` if overflowing memory; same goes for the following two steps)
   ```bash
   python scripts/train_direct.py
   ```
7. Run the following command to fine-tune the model with LoRA:
   ```bash
   python scripts/train_lora.py
   ```
8. Run the following command to evaluate the model:
   ```bash
   python scripts/evaluate.py
   ```
   use `--model <direct/lora>` to specify the model to evaluate. by default both models will be evaluated.
   The plots will be saved to `plots/`. A `.md` file containing all graph information would also be available there.
## Pipeline Overview

- **Data Processing:**
  - Load and preprocess data, add index, encode labels, split into train/test, and tokenize.
- **Model Training:**
  - Train RoBERTa with direct fine-tuning (`scripts/train_direct.py`).
  - Train RoBERTa with LoRA adapter (`scripts/train_lora.py`).
- **Evaluation:**
  - Evaluate both models, plot confusion matrices, and compute/rank F1 scores per topic (`scripts/evaluate.py`).

## Test-Driven Development

All modules are developed using TDD. See the `tests/` directory for test cases.

## Notes
- Requires a GPU for efficient training.
- For configuration, you may use `config.yaml` (optional).

## License
MIT 