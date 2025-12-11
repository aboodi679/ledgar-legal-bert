# Legal Document Classification - Legal-BERT

🏛️ **Automated classification of legal documents into 100 provision types using a fine-tuned Legal-BERT model**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Project Overview

This project implements a complete NLP pipeline for classifying legal documents from the LEDGAR dataset using a domain-specific Legal-BERT model. The system achieves **84% accuracy** on test data and recognizes 100 different legal provision types.

### Key Features
- ✅ Complete data preprocessing pipeline with text cleaning
- ✅ Multiple embedding techniques (Word2Vec, GloVe, FastText)
- ✅ Fine-tuned Legal-BERT transformer model
- ✅ Interactive Gradio web interface for inference
- ✅ Comprehensive evaluation metrics and visualizations

---

## 🎯 Performance Metrics

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 84.0% |
| **Macro F1-Score** | 74.15% |
| **Macro Precision** | 76.59% |
| **Macro Recall** | 74.26% |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/aboodi679/ledgar-legal-bert.git
cd ledgar-legal-bert
pip install -r requirements.txt
```

### 2. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. Run the Web App
```bash
python app.py
```

---

## 📁 Project Structure
```
ledgar-legal-bert/
├── notebook/              # Google notebooks
│    Complete Notebook.ipynb
│   
│   
├── src/                    # Source code modules
│   ├── preprocess.py       # Data cleaning functions
│   ├── train.py            # Model training
│   ├── inference.py        # Prediction functions
│   └── utils.py            # Helper functions
├── app.py                  # Gradio web interface
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🔧 Pipeline Stages

### 1. Data Preprocessing
- Load LEDGAR dataset from Hugging Face
- Text cleaning (lowercase, remove special chars, lemmatization)
- Remove stopwords and normalize whitespace
- Generate TF-IDF, Bag-of-Words representations
- Create embeddings (Word2Vec, GloVe, FastText)

### 2. Model Training
- **Model**: `nlpaueb/legal-bert-base-uncased`
- **Training**: 3 epochs with AdamW optimizer
- **Batch Size**: 8 (effective 16 with gradient accumulation)
- **Learning Rate**: 3e-5 with linear warmup
- **Mixed Precision**: FP16 for faster training

### 3. Inference
- Web-based interface using Gradio
- Returns top-5 predictions with confidence scores
- Maps numeric IDs to human-readable provision names

---

## 📈 Results & Visualizations

![Training Loss](path/to/training_loss.png)
![Class Distribution](path/to/class_distribution.png)

*(Add your generated graphs here)*

---

## 🛠️ Usage Examples

### Command Line Inference
```python
from src.inference import predict_legal_provision

text = "This Agreement shall be governed by the laws of Delaware."
result = predict_legal_provision(text)
print(result['top_prediction'])
# Output: {'name': 'Governing Law', 'confidence': 0.95}
```

### Web Interface
Run `python app.py` and visit the provided URL.

---

## 📚 Dataset

**LEDGAR** - Legal Document Classification Dataset
- **Source**: [Hugging Face - lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue)
- **Size**: 60,000+ legal contract clauses
- **Classes**: 100 provision types
- **Domain**: Contract law

---

## 🧠 Model Architecture

- **Base Model**: Legal-BERT (BERT-base pre-trained on legal corpus)
- **Parameters**: ~110M
- **Layers**: 12 transformer layers
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Max Sequence Length**: 512 tokens

---

## ⚡ Optimizations

- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)
- Gradient accumulation
- Dynamic padding
- Cached tokenization

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) by NLP-AUEB
- [LEDGAR Dataset](https://huggingface.co/datasets/coastalcph/lex_glue) from LexGLUE benchmark
- Hugging Face Transformers library

---

## 📧 Contact

For questions or feedback, please open an issue or contact me at aaboodi679@gmail.com

---

**⭐ If you find this project helpful, please give it a star!**
