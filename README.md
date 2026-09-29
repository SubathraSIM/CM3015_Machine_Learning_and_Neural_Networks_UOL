# CM3015 Machine Learning and Neural Networks - University of London

Coursework for **CM3015 Machine Learning and Neural Networks** (BSc Computer Science, University of London). The midterm compares classical supervised learning algorithms, including a KNN written from scratch. The endterm follows Chollet's universal deep learning workflow to build, overfit, regularise, and tune a neural network.

| Assessment | Project | Key topics |
|---|---|---|
| Mid Term | Supervised models and PCA on the Wine dataset | KNN from scratch, Decision Tree, Naive Bayes, cross-validation, PCA |
| End Term | IMDB sentiment classification with neural networks | Keras, overfitting, L2 and dropout regularisation, hyperparameter tuning |

**Tech:** Python · NumPy · scikit-learn · TensorFlow / Keras · Matplotlib · Jupyter

---

## Midterm - Comparing Supervised Models and the Impact of PCA

Three classifiers with very different inductive biases are compared on the classic **Wine dataset** (13 chemical features, 3 cultivars):

- **K-Nearest Neighbours:** implemented **from scratch** in Python with Euclidean distance and majority voting
- **Decision Tree:** rule-based learner (scikit-learn)
- **Gaussian Naive Bayes:** probabilistic baseline (scikit-learn)

### Method
- EDA: class balance, feature distributions, correlation heatmap
- Standardisation and an 80/20 train-test split
- Evaluation with accuracy, confusion matrices, classification reports, and learning curves
- **Stratified 5-fold cross-validation** to test stability
- **PCA** reduction to 2 components to test how the models hold up with fewer features

**Takeaways:** Naive Bayes was the most accurate and the most stable. The Decision Tree varied the most between folds. Cutting 13 features down to 2 principal components lowered accuracy for every model, but performance stayed competitive.

---

## Endterm - IMDB Sentiment Classification (Universal DL Workflow)

Binary sentiment classification of **50,000 IMDB movie reviews** (25k train / 25k test, perfectly balanced). The project works through each step of the universal machine learning workflow from *Deep Learning with Python* (Chollet).

### Workflow
1. **Define the problem and success measure:** accuracy (balanced classes), trained with binary cross-entropy
2. **Evaluation protocol:** hold-out validation (5,000 reviews)
3. **Data preparation:** reviews converted to 10,000-dimension multi-hot bag-of-words vectors
4. **Baseline:** small dense network (8-8-8) to prove the problem is learnable
5. **Deliberate overfitting:** deep network (512-256-128-64-32, 5.3M parameters) trained to 100% training accuracy
6. **Regularisation:** L2 only, Dropout only, and L2 + Dropout combined
7. **Hyperparameter tuning:** three variations of the dropout rate and L2 strength

**Takeaways:**
- The overfit model showed the classic pattern: training loss fell to almost zero while validation loss rose past 1.3.
- Dropout gave the best raw accuracy but poorly calibrated, overconfident predictions (high loss).
- **Combining L2 and dropout** gave the best balance of accuracy and stable loss, and tuning the L2 strength down to 0.0005 gave the highest validation accuracy.

---

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow jupyter
jupyter notebook
```
- **Midterm:** the Wine dataset loads from `sklearn.datasets`, so no download is needed.
- **Endterm:** open `ML_End_Term.ipynb`. IMDB downloads automatically through `keras.datasets`.
