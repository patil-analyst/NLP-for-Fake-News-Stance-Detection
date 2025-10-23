# Natural Language Analysis: Claim Relatedness Classification

## 1. Project Overview

This repository contains the Jupyter Notebook (`Code_NLP.ipynb`) for a binary text classification task: determining if a news "Claim" is **Related** or **Unrelated** to an accompanying "Article".

This project was completed as an assessment for the Natural Language Analysis (NLA) module at Durham University. The notebook provides the complete implementation, feature extraction, model training, and evaluation.

For a detailed analysis and interpretation of results, please refer to the main assessment report (`[Your_Report_Name].pdf`) submitted separately via Ultra.

## 2. Key Skills & Techniques

This project demonstrates a range of skills in data science, machine learning, and natural language processing:

* **Data Science:** Data loading, cleaning, and preprocessing using `pandas`.
* **Natural Language Processing (NLP):** Binary text classification, 4-way classification, and advanced feature extraction.
* **Machine Learning:**
  * **Classical Models:** Implementing and evaluating traditional ML pipelines using `TF-IDF` with `Logistic Regression` and `SVM`.
  * **Hybrid Models:** Using `Transformer Embeddings` as features for a `Logistic Regression` classifier.
* **Deep Learning & Generative AI:**
  * **Zero-Shot Classification:** Applying pre-trained `BERT-MNLI` models directly to the task.
  * **Few-Shot Learning:** Prompting generative models like `GPT-2`.
  * **Advanced Prompt Engineering:** Implementing a **Chain-of-Thought (CoT)** strategy with `FLAN-T5` to improve reasoning and classification accuracy.
* **Model Evaluation:** Quantitative analysis using `scikit-learn` (Accuracy, Precision, Recall, F1-Score).
* **Toolkit:** Python, Jupyter Notebooks, PyTorch, Transformers, `scikit-learn`, `pandas`.

## 3. File Contents

* `Code_NLP.ipynb`: The main Jupyter Notebook containing all Python code and experimental results.
* `README.md`: This file, providing setup and execution instructions.
* *(Data Files)*: The notebook assumes the two data files (`claims.csv`, `articles.csv`) are present in the same directory.
