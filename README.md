# AI-NER-spaCy

# Custom Named Entity Recognition (NER) with spaCy

This repository contains a demo of training a Named Entity Recognition (NER) model using [spaCy](https://spacy.io/) to identify quantities and product names in text. The model is trained on custom data containing queries about purchasing products.

## 📌 What is Named Entity Recognition (NER)?
Named Entity Recognition (NER) is a Natural Language Processing (NLP) technique that identifies and categorizes key entities in text. Common entity types include names, dates, organizations, and locations. In this demo, we train a custom model to recognize:
- **QUANTITY** – Number of items being requested (e.g., "10" in "10 bananas").
- **PRODUCT** – The product being referenced (e.g., "bananas" in "10 bananas").

## 🛠 Technologies Used
- **Python**: Programming language for NLP processing.
- **spaCy**: Open-source NLP library used for NER training and inference.
- **Thinc**: Machine learning framework used internally by spaCy.

## 📜 Code Overview
1. **Training Data** – A set of sample purchase-related queries with labeled entities (`QUANTITY` and `PRODUCT`).
2. **Model Preparation** – Loads a pre-trained spaCy model (`en_core_web_md`) and configures the NER pipeline.
3. **Training** – Iterates over the dataset for 50 epochs, updating the model with new entity labels.
4. **Saving and Loading the Model** – Saves the trained model to disk and reloads it for testing.
5. **Testing** – Runs test sentences through the trained model to extract named entities.

## 🚀 How to Run the Code
### 1️⃣ Install Dependencies
Ensure you have Python installed, then install the required packages:
```bash
pip install spacy
python -m spacy download en_core_web_md
