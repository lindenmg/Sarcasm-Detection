# Sarcasm Detection Project

This was a project for the 2017/18 winter semester university lecture/workshop course *Deep Learning for Speech and Language Processing*.
It was conducted by Gabriel Lindenmaier and Pascal Weiss from the University of Stuttgart. 
Our chosen project was to develop a system capable of detecting sarcasm in web forum comments using Deep Neural Networks.

We built classifiers using Deep Neural Networks to process sarcastic comments from the [SARC corpus](https://paperswithcode.com/dataset/sarc),
which contains Reddit posts. Our CNN achieved human-level performance despite noisy data.
An unoptimized convolution with attention performed slightly better than normal convolution
and served as a good feature generator for other Machine Learning (ML) classifiers.

The primary task was to classify replies as sarcastic or not using Deep Learning.
PyTorch was used to implement the Neural Networks. The more in-depth going research question
of Gabriel Lindenmaier focused on combining convolution layers with attention
and using the CNN as a feature generator for other ML classifiers.
The research question of Pascal Weiss was about creating a Bayesian hyperparameter optimization
and comparing it with grid search and random search.

## Table of Contents
1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Data Preparation](#data-preparation)
4. [Training Pipeline](#training-pipeline)
5. [Research Questions](#research-questions)
   - [Gabriel's Research Question](#research-question-gabriel)
6. [Results and Analysis](#results-and-analysis)
7. [Conclusion](#conclusion)
8. [Logging and Monitoring](#logging-and-monitoring)
9. [Miscellaneous](#miscellaneous)

## Overview

**File Overview:**
```
├── config.json_template <== The template for the config.json
├── data                 <== Where you have to store the pure data (already contains stop word files)
├── LICENCE
├── README.md            <== The file you are reading
├── requirements.txt     <== What you have to install with pip install -r
├── src                  <== The code base
├── start_mongodb.sh     <== Script to start the database for logging
└── test                 <== The (unit) tests for the code in src/
```

**src/ Code Base:**
```
├── preprocessing      <== Data handling, preprocessing
├── data_science       <== Classes around DNN setup and execution
├── hyperparameters    <== Hyperparameter search
├── loggers            <== Logging classes
├── models             <== DNN models
├── playground_gabriel <== Experimentation, data exploration, research question of Gabriel
├── playground_pascal  <== Experimentation of Pascal
├── strategies         <== Interface implementation for training pipeline (research question Pascal)
├── tools              <== Database interface for logging, metrics, utilities
└── training           <== Training & preprocessing pipeline with caching
```

## Setup Instructions

### Python Environment
- Python 3.6 is recommended
- Using a virtual environment is recommended.
- Install PyTorch manually to choose the correct version for your system. The CPU version is recommended.
  (refer to requirements.txt for the required version numbers)
- Install the Python requirements with `pip install -r requirements.txt`.
- If issues arise with 'thinc' and 'spacy' packages, first install 'thinc' manually in the given version, then 'spacy'.

### Miscellaneous Tools
- We use PyTest and Hypothesis for unit tests.
- For spaCy, after installation, activate your virtual environment and run:
  ```
  python -m spacy download en_core_web_md
  python -m spacy download en
  ```

### Config File
- Copy `config.json_template` and rename the copy to `config.json`. Add the corresponding values to each key in the file that fits your desired configuration.
- For backward compatibility of some old Jupyter notebooks, the path defined in `data_folder` should be `[project root path]/data/`.

## Data Preparation
For reference: You can find the full original [dataset](https://huggingface.co/datasets/CreativeLang/SARC_Sarcasm) currently on Hugging Face.

### Data Cleaning
This **was** for pre-processing the dataset provided for the course. This repository comes 
with already processed datasets, therefore this section remains empty.

### Training & Test Data Setup
Copy `comments_cleaned.txt`, `annotation.txt`, `train.csv`, `test.csv` into `data/` & `test/test_data`
After running unit tests you might have to recopy the data files.

### Word Vectors
Store word vectors in the `data` folder as follows:
```
├── data
│   └── word_vectors
│       ├── fastText
│       │   ├── crawl-300d-2M.vec
```
Download the word vectors from [here](https://fasttext.cc/docs/en/english-vectors.html) and convert them using `src/tools/rewrite_word_vectors.py`.

## Training Pipeline

To execute the preprocessing, training, and test pipeline, use the implementations in `src/strategies/`. 
For the CNN, use `src/strategies/cnn/exec_cnn_baseline.py` with the appropriate configuration files. 
Important configuration parameters include:
The important parts of the config files for the pipeline works as follows:
```
  "learning_session": {
    "epochs": 7,                  <== For how many epochs you want to run the session
    "cnn_mode": true,             <== true if you use CNN classes, false else
    "cache_prefix": "cnn_simple", <== For the caching of preprocessing. If you change this the caches get another name.
    "mode": "train",              <== The mode of the pipeline execution. "train" for k-fold cross validation, "tr_full" for training on the full training set and "test" for testing on the test set. Modes can also be combined. The order of execution is always train-tr_full-test
    "mode_to_load": "train",      <== If the mode is "test" this defines what stored model it should load. The last model saved under the given mode
    "fold_to_load": 9,            <== If you want to load a model from "train" mode you can choose the fold of the CV. Use 1 as value in case of "tr_full"
    "save_interval": 7,           <== After which epoch the model is saved in "train" and "tr_full" mode
    "seed": 87076143,
    "session_tag": "cnn_baseline" <== Tag for logging
  },
```

```
"data_factory": {
    "args": {
      [..]
        "test_file": "test.csv",   <== The name of the test file to use in data/
      [..]
        "vocab": {
          "word_vector_path": "word_vectors/fastText/ft_2M_300.csv",
          "word_vec_count": 2000000,   <== Number of vectors in the word vector file
          "word_vec_dim": 300,         <== Word vector dimension
          "ft_format": false,          <== true if the word vector file has a header in the format <vector count> <vector dim>
          [..]
```

## Research Questions

### Research Question Gabriel
- Save a CNN model using `src/strategies/cnn/execute_cnn_baseline.py` with `attentive_cnn_config.json`.
- Extract features using `src/playground_gabriel/feature_extraction.ipynb`.
- Train ML classifiers using `src/playground_gabriel/research_question.ipynb`.

## Results and Analysis

- Baseline CNN achieved ~67.6% accuracy.
- CNN with attention achieved ~67.82% accuracy (with only slightly changed hyperparameters compared to baseline).
- ML classifiers based on attention CNN features achieved up to 68.64% accuracy individually and 69.1% as an ensemble.
- A survey comparing human performance showed the CNN performs comparably to humans under same circumstances.

## Conclusion

The project demonstrated that deep learning models could effectively detect sarcasm in text. 
Future work could have focused on optimizing the attention mechanism 
and improving data quality by retaining links and annotating non-sarcastic posts better.

## Logging and Monitoring

### Tensorboard Logging
- Define the path to your logs in `config.json` with the key `path.log_folder`.
- To run and access Tensorboard on a remote machine:
  ```bash
  tensorboard --port <your favourite port> --logdir <your logdir>
  ssh -N -f -L localhost:8888:localhost:<your port> <username>@<remote address>
  ```

### MongoDB Logging
- Follow the [MongoDB Installation Instructions](https://www.mongodb.com/docs/manual/installation/) for your OS.
- Start MongoDB with `start_mongodb.sh`.

## Miscellaneous

### Execute Python From Everywhere
To execute a `.py` file that imports a module in a lower hierarchy, add the project root to `sys.path` before your import statements:
```python
import os
import sys
project_root_path = os.path.realpath(os.path.join(__file__, "../", "..", ".."))
sys.path.append(project_root_path)
import src.tools.helpers
```
