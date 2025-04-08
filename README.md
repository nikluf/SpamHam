# SpamHam - SMS Spam Classifier

**SpamHam** is a machine learning project that detects spam messages using natural language processing (NLP) and a Naive Bayes classifier. This tool is trained on a classic SMS spam dataset and can predict whether a given message is **Spam** or **Ham** (not spam).
## Overview

- Trains on a SMS spam dataset
- Cleans and vectorizes text using TF-IDF
- Uses a Naive Bayes model for classification
- CLI-based interface for testing predictions

---

## Files

- `pckbuilder.py` - Builds and saves the model and vectorizer
- `tester.py` - Loads the model and predicts new messages
- `requirements.txt` - Required Python packages
- `spam.csv` - SMS Spam Collection dataset used for training ([Source](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download))
---
## Quick Start

```bash
pip install -r requirements.txt
python3 pckbuilder.py
python3 tester.py




Input: You've won a prize! Click here now!
Output: Spam

Input: Hey, are we still meeting later?
Output: Not Spam
