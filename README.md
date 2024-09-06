# Resume Classification using NLP and Deep Learning
## Project Overview
This project aims to develop a deep learning model that can automatically classify resumes into different job categories based on the content of the resume. The model uses Natural Language Processing (NLP) techniques and is trained on a large dataset of resumes.

## Dataset
The project uses four datasets:

* HuggingFace Dataset 1: dataset from HuggingFace
* HuggingFace Dataset 2: dataset from HuggingFace
* CSV Dataset 1: dataset from Kaggle
* CSV Dataset 2: dataset from GitHub

## Data Preprocessing
The datasets are preprocessed by:

* Removing punctuation and extra spaces
* Converting all text to lowercase
* Tokenizing the text using the BERT tokenizer

## Model
The project uses a `BERT-based` sequence classification model, specifically the `BertForSequenceClassification` model from the `Hugging Face Transformers` library.

## Training
The model is trained on the preprocessed dataset using the Trainer API from the Hugging Face Transformers library. The training process involves fine-tuning the `pre-trained BERT` model on the resume classification task.

## Evaluation
The model is evaluated on a test set using metrics such as `accuracy, precision, recall, and F1-score`.

## Pipeline
The project creates a pipeline for resume classification using the trained model and the `BERT tokenizer`. The pipeline takes in a resume text as input and outputs the predicted job category and score.

## Testing
The pipeline is tested on several resume PDF files using the test_resume function, which extracts text from the PDF files, preprocesses the text, and passes it through the pipeline to get the predicted job category and score.

## Usage
To use the pipeline, simply run the `test_resume` function and pass in the path to a resume PDF file as an argument.

## License
This project is licensed under the MIT License.
