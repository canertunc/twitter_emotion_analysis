# 🎭 Twitter Emotion Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![BERT](https://img.shields.io/badge/BERT-Transformer-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)

## 📌 Project Overview

This project performs sentiment classification and emotion analysis on Twitter data. The dataset consists of 416,809 tweets with 6 different emotion labels (0-5), representing Sadness, Happiness, Love, Anger, Fear, and Surprise.

### 🎯 Project Goals

- Implement a complete machine learning pipeline for emotion classification
- Compare performance of traditional ML models and transformer-based approaches
- Handle class imbalance and evaluate model performance
- Extract meaningful insights from Twitter text data

## 📊 Dataset

The Twitter Emotion Classification Dataset contains 416,809 tweets labeled with the following emotions:
- 0: Sadness (29.07%)
- 1: Happiness (33.84%)
- 2: Love (8.29%)
- 3: Anger (13.75%)
- 4: Fear (11.45%)
- 5: Surprise (3.59%)

Source: [Twitter Emotion Classification Dataset on Kaggle](https://www.kaggle.com/datasets/aadyasingh55/twitter-emotion-classification-dataset/data)

## 🔄 Project Workflow

### 1. Data Loading and Inspection
- Dataset shape analysis: 416,809 tweets with 2 columns (text, label)
- Examination of class distribution
- Checking for null values (none found)

### 2. Data Cleaning and Preprocessing
- Text cleaning functions to remove:
  - URLs, mentions, hashtags
  - HTML tags, emojis, special characters
  - Punctuation, numbers
- Text normalization:
  - Conversion to lowercase
  - Stopword removal
  - Lemmatization and stemming

### 3. Exploratory Data Analysis (EDA)
- Class distribution visualization
- Text length analysis
- Word clouds for different emotions
- Most frequent words analysis
- Sentiment score distribution

### 4. Feature Extraction
- TF-IDF (Term Frequency-Inverse Document Frequency)
  - Directly represents text content by considering word frequencies
  - Emphasizes importance of words within documents and across the dataset
  - Allows models to learn more distinct discriminative features
- Word2Vec embeddings
  - Represents semantic similarities between words
  - Context-dependent nature of vectors can sometimes limit performance
  - More sensitive to dataset-specific variations

Both TF-IDF and Word2Vec were evaluated, with TF-IDF generally providing better results for the machine learning models. The performance metrics table primarily shows results from models trained with TF-IDF features.

### 5. Data Balancing
- Addressing class imbalance using undersampling
  - Reduces the number of examples from majority classes
  - Creates a more balanced distribution across all emotion categories
  - Improves learning performance by preventing models from biasing toward majority classes
- Analysis of balanced vs. imbalanced training approaches

### 6. Modeling

#### 📝 Traditional ML Models
- Decision Tree
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes
- XGBoost

#### 🤖 Deep Learning Models
- BERT (Bidirectional Encoder Representations from Transformers)

### 7. Performance Evaluation
- Accuracy, F1-Score
- Confusion matrix analysis
- Cross-validation
- Model comparisons

## 🚀 Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **NLTK & Regex**: Text preprocessing
- **Scikit-learn**: Traditional ML models and evaluation
- **TensorFlow & PyTorch**: Deep learning implementations
- **Transformers**: BERT implementation
- **WordCloud**: Text visualization
- **imbalanced-learn**: Handling class imbalance

## 🔍 Key Findings

- Class imbalance significantly affects model performance
- BERT outperforms traditional ML models in emotion classification
- Different emotions have distinct linguistic patterns
- Text preprocessing plays a crucial role in model performance
- Balancing methods improve results for minority classes

## 📈 Results

The project compares various models for emotion classification with the following key metrics:

| Model | Accuracy | F1-Score |
|-------|---------|---------|
| Decision Tree | 87% | 87% |
| Logistic Regression | 90% | 91% |
| Random Forest | 91% | 91% |
| KNN | 40% | 40% |
| Naive Bayes | 88% | 89% |
| XGBoost | 89% | 90% |
| **BERT** | **93%** | **93%** |

As shown in the table, **BERT** significantly outperforms traditional machine learning models in emotion classification tasks, achieving superior accuracy and F1-scores.

## 📝 Project Structure

The entire project is contained in a single Jupyter notebook (twitter_emotion_analysis.ipynb) with the following structure:
1. Introduction and problem definition
2. Data loading and preprocessing
3. Exploratory data analysis
4. Feature extraction
5. Model implementation and evaluation
6. Conclusion

## 💡 Areas for Improvement

1. **Increasing the Number of Epochs**: The model was trained with 2 epochs, but analysis of metrics and learning curves indicated this was insufficient for full convergence. Increasing to 4-5 epochs could significantly improve performance.

2. **Hyperparameter Optimization**: Optimizing BERT hyperparameters (learning rate, batch size, sequence length) through grid search or random search methods. While some models used random search, applying more extensive hyperparameter tuning across all models would yield better results.

3. **Comparison of Different Transformer-Based Models**: Train models with alternative transformer architectures like RoBERTa, DistilBERT, and XLNet to determine the most suitable transformer model for this task.

4. **Hybrid Data Balancing Approach**: This study used only undersampling. A hybrid approach combining undersampling with oversampling methods (like SMOTE) could potentially improve performance for minority classes.

5. **Spell Checking**: Implementing spell correction could improve results, especially for word-based vectorization methods like TF-IDF, but was not performed due to time constraints.

6. **Exploring and Optimizing Word Embeddings**: Experiment with different word embedding techniques and optimize Word2Vec models more thoroughly to achieve better results.

## 📑 Detailed Report

For a more comprehensive analysis and detailed discussion of methods, results, and conclusions, please refer to the `report.pdf` file available in this GitHub repository. 

## 🏆 About This Project

This project was created as a pre-internship task assigned by IdaMobile company. It was developed as part of the application process for a volunteer internship position and demonstrates a complete machine learning solution for emotion analysis on Twitter data. 
