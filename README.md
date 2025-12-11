# **Legal Document Classification – NLP Project (BERT + Web App)**

This repository contains my full NLP pipeline for classifying legal documents using a fine-tuned **Legal-BERT** model.
It includes **dataset preprocessing, text representation, embeddings, model training, evaluation, and a complete Gradio web app** for inference.

---

## 🚀 **Project Overview**

The goal of this project is to automatically classify legal text into its correct **provision/category**.
This helps in quickly identifying the type of legal document without reading the full text.

The pipeline consists of:

1. **Dataset preprocessing**
2. **Text cleaning**
3. **Tokenization**
4. **Embedding generation**
5. **Model building (Legal-BERT)**
6. **Training & validation**
7. **Saving model + label encoder**
8. **Gradio-based Web App for prediction**

---

## 🧹 **1. Dataset Preprocessing**

* Removed null values
* Cleaned text
* Normalized whitespace
* Lowercased text
* Prepared target labels
* Split dataset into **train / test**

---

## 🔠 **2. Text Representation & Embedding**

The model uses **BERT embeddings** internally:

* Tokenization via `AutoTokenizer`
* Conversion to input IDs & attention masks
* BERT converts each token into a **768-dimensional embedding**
* These embeddings represent the meaning of the text

---

## 🤖 **3. Model Architecture**

Using **AutoModelForSequenceClassification (Legal-BERT)**:

### **Model Layers**

* Embedding layer
* 12 Transformer encoder layers
* Self-attention mechanism
* Classification head (Dense layer)
* Softmax for output probabilities

---

## 🏋️ **4. Training**

* Loss: CrossEntropy
* Optimizer: AdamW
* Batch size: 16
* Epochs: (customizable)
* Validation evaluation

### **Overfitting / Underfitting**

* Overfitting → high train accuracy, low test accuracy
* Underfitting → poor accuracy on both

---

## 🧪 **5. Inference + Web App**

The web app allows users to:

* Input legal text
* Preprocess it using the same pipeline
* Use the trained model to predict category
* Displays actual provision name using saved LabelEncoder

---

## 🎛 **6. How to Run Locally**

### **Install dependencies**

```bash
pip install -r requirements.txt
```

### **Run the Gradio App**

```bash
python app.py
```

Or open `inference_webapp.ipynb` in Colab.

---

## 📈 **Results**

* Achieved good accuracy on validation set
* Model generalizes well on unseen legal text
* Works best when text is long and detailed

---

## 🚫 **Limitations**

* Cannot classify every legal document—depends on training data
* Very short sentences reduce accuracy
* Domain-specific vocabulary improves performance

---

## 🧾 **License**

MIT License — free to use & modify.

---
