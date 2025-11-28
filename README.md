# Fake News Detection

A machine learning project for detecting fake news using NLP techniques, TF-IDF vectorization, and machine learning classifiers.

## Features

- **Text Preprocessing**: Clean and normalize text data (remove URLs, HTML, punctuation, stopwords, and apply stemming)
- **TF-IDF Vectorization**: Convert text to numerical features using Term Frequency-Inverse Document Frequency
- **Multiple Classifiers**: Support for Logistic Regression, Naive Bayes, and SVM
- **Model Persistence**: Save and load trained models
- **Flask API**: REST API for predictions
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
fake-news-detection/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # Text preprocessing utilities
│   ├── train.py           # Model training and classifier
│   ├── evaluate.py        # Evaluation metrics
│   └── app.py             # Flask API
├── tests/
│   ├── test_preprocessing.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_app.py
├── data/
│   └── sample_data.csv    # Sample dataset
├── models/                 # Saved models (gitignored)
├── requirements.txt
└── README.md
```

## Usage

### Training a Model

```python
from src.train import FakeNewsClassifier, train_model

# Train from CSV file
classifier, X_train, X_test, y_train, y_test = train_model(
    'data/sample_data.csv',
    text_column='text',
    label_column='label',
    save_path='models/model.pkl'
)

# Or train manually
classifier = FakeNewsClassifier(classifier_type='logistic_regression')
classifier.train(texts, labels)
classifier.save('models/model.pkl')
```

### Making Predictions

```python
from src.train import FakeNewsClassifier

# Load model
classifier = FakeNewsClassifier.load('models/model.pkl')

# Predict
prediction = classifier.predict("Breaking news about science")
print(prediction)  # 0 = real, 1 = fake
```

### Running the Flask API

```bash
cd src
python app.py
```

The API will be available at `http://localhost:5000`.

#### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict single article
  ```json
  {"text": "Your news article text here"}
  ```
- `POST /batch_predict` - Predict multiple articles
  ```json
  {"texts": ["Article 1", "Article 2"]}
  ```

### Evaluation

```python
from src.evaluate import evaluate_model, print_evaluation

# Evaluate predictions
y_pred = classifier.predict(X_test)
metrics = evaluate_model(y_test, y_pred)
print_evaluation(y_test, y_pred)
```

## Running Tests

```bash
pytest tests/ -v
```

## Classifier Options

- `logistic_regression` (default) - Good balance of speed and accuracy
- `naive_bayes` - Fast training, works well with text data
- `svm` - Good for high-dimensional data

## License

MIT License

