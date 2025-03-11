import random

import spacy
from spacy.util import minibatch
from spacy.training.example import Example
from thinc import optimizers

# Training data with labeled entities (QUANTITY and PRODUCT)
train_data = [
    ("What is the price of 10 bananas?", {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]}),
    ("What is the price of 5 laptops?", {"entities": [(21, 23, "QUANTITY"), (23, 30, "PRODUCT")]}),
    ("How much are 7 bottles?", {"entities": [(13, 14, "QUANTITY"), (15, 22, "PRODUCT")]}),
    # Additional samples for better model generalization
]

# Load a pre-trained spaCy model
nlp = spacy.load('en_core_web_md')

# Add the Named Entity Recognizer (NER) pipeline if not already present
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')
else:
    ner = nlp.get_pipe('ner')

# Add new entity labels (QUANTITY and PRODUCT) to the NER model
for _, annotations in train_data:
    for ent in annotations['entities']:
        if ent[2] not in ner.labels:
            ner.add_label(ent[2])

# Disable other pipeline components to focus only on NER training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    # Train the model for 50 epochs
    epochs = 50
    for epoch in range(epochs):
        random.shuffle(train_data)  # Shuffle training data each epoch for better generalization
        losses = {}
        batches = minibatch(train_data, size=2)  # Train in small batches
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)  # Convert text into spaCy Doc object
                example = Example.from_dict(doc, annotations)  # Create Example object
                examples.append(example)
            nlp.update(examples, drop=0.5, losses=losses)  # Update model with dropout for better generalization
        print(f'Epoch {epoch + 1}, Losses: {losses}')

# Save the trained model to disk
nlp.to_disk('custom_ner_model')

# Load the trained model for testing
trained_nlp = spacy.load('custom_ner_model')

# Sample test texts
test_texts = [
    "How much for 3 oranges?",
    "I want 15 chairs for the conference.",
    "Can you give me the price for 6 desks?"
]

# Run NER on test texts
for text in test_texts:
    doc = trained_nlp(text)
    print(f'Text: {text}')
    print('Entities:', [(ent.text, ent.label_) for ent in doc.ents])
    print()

