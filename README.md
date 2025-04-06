# Sentiment Analysis of Customer Reviews

This project analyzes customer reviews from e-commerce platforms to determine sentiment polarity (positive, negative, neutral) using Natural Language Processing and Machine Learning techniques.

## Features

- Web scraping of customer reviews
- Data preprocessing and cleaning
- Sentiment analysis using ML algorithms
- Interactive data visualization
- Web application for real-time sentiment analysis

## Project Structure

```
sentiment_analysis/
├── data/                  # Data storage
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code
│   ├── data/            # Data collection and preprocessing
│   ├── models/          # Model training and evaluation
│   ├── visualization/   # Data visualization
│   └── web/             # Flask web application
├── tests/               # Unit tests
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Data Collection:
   ```bash
   python src/data/scraper.py
   ```

2. Model Training:
   ```bash
   python src/models/train.py
   ```

3. Run Web Application:
   ```bash
   python src/web/app.py
   ```

## Technologies Used

- Python 3.8+
- Beautiful Soup for web scraping
- Pandas & NumPy for data manipulation
- Scikit-learn for ML algorithms
- NLTK for NLP tasks
- Matplotlib & Seaborn for visualization
- Flask for web application

## License

MIT License 