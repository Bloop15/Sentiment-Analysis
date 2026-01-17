# Sentiment Analysis using LSTM and NLP

This project demonstrates a complete **Natural Language Processing (NLP)** workflow for **binary sentiment classification** of movie reviews using the **IMDb dataset**. The objective is to compare a strong classical NLP baseline with deep learning models and analyze how different modeling choices affect performance.

---

## Dataset Description and Preprocessing

- **Dataset:** IMDb Movie Reviews (50,000 reviews)
- **Task:** Binary sentiment classification (positive / negative)
- The dataset is balanced with equal positive and negative samples.

### Preprocessing Steps
- Reviews are converted into integer sequences using a predefined word index.
- Vocabulary size is restricted to the top **10,000 most frequent words**.
- Sequences are padded or truncated to a fixed maximum length.
- Padding tokens are masked to prevent them from influencing model learning.
- Pre-trained **GloVe (100-dimensional)** embeddings are used to incorporate semantic information.

---

## Modeling Approach

A systematic modeling pipeline was followed to ensure fair comparison and robust evaluation:

### Baseline Model
- **TF-IDF vectorization**
- **Logistic Regression classifier**
- Serves as a strong classical NLP baseline.

### Deep Learning Models
- **Embedding → LSTM → Dense** architecture
- Pre-trained GloVe embeddings used for word representation
- Experiments conducted with:
  - Frozen (non-trainable) embeddings
  - Trainable embeddings
  - EarlyStopping for regularization
  - Bidirectional LSTM (exploratory)

The final deep learning model uses an **LSTM with trainable GloVe embeddings** and EarlyStopping.

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| TF-IDF + Logistic Regression | 0.884 | 0.88 | 0.88 | 0.88 |
| LSTM + Trainable GloVe Embeddings | 0.884 | 0.88 | 0.88 | 0.88 |

---

## Training Analysis

- Training and validation accuracy increase rapidly during initial epochs.
- Validation performance plateaus after a few epochs, indicating the onset of overfitting.
- **EarlyStopping** effectively selects the optimal training point.
- Confusion matrices show balanced predictions across both sentiment classes.

---

## Conclusion

The final LSTM model with trainable GloVe embeddings achieves performance **comparable to the strong TF-IDF + Logistic Regression baseline**. While deep learning does not significantly outperform the classical approach in terms of accuracy, it successfully matches baseline performance while learning richer contextual and semantic representations.

This project highlights an important insight: **strong classical NLP models can remain highly competitive**, and deep learning models require careful representation learning and tuning to justify increased complexity.

---

## Future Work

Potential extensions include:
- Transformer-based models such as **BERT**
- Attention mechanisms for improved sequence modeling
- Error analysis on misclassified reviews
