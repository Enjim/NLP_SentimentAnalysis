import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import seaborn as sns
from langdetect import detect
from collections import Counter
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "Error"  # In case the language detection fails

# Load the dataset
df = pd.read_csv('data.csv', delimiter=',')
df.columns = ['text', 'sentiment']  # If the columns are not named in your csv

actual_text_column_name = 'text'
df['language'] = df[actual_text_column_name].apply(detect_language)
df= df[df['language'] == 'en']


# Map sentiments to integers
encoder = LabelEncoder()
df['sentiment'] = encoder.fit_transform(df['sentiment'])

#In order to find out the mapping of the encoding to the labels
label_mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))

# Split your dataset into train and test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=123)

# Split dataframes into x (input) and y (output)
train_text = train_df['text'].to_list()
train_labels = train_df['sentiment'].to_list()
test_text = test_df['text'].to_list()
test_labels = test_df['sentiment'].to_list()

# Tokenize the text
tokenizer = Tokenizer(num_words=400, oov_token='<OOV>')
tokenizer.fit_on_texts(train_text)

# Convert texts to sequences of integers
train_sequences = tokenizer.texts_to_sequences(train_text)
test_sequences = tokenizer.texts_to_sequences(test_text)

# Pad the sequences so they're all the same length
train_padded = pad_sequences(train_sequences, padding='post')
test_padded = pad_sequences(test_sequences, padding='post')

# Convert to numpy arrays
train_padded = np.array(train_padded)
test_padded = np.array(test_padded)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


# Adding the model checkpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Model definition
model = tf.keras.Sequential()
model.add(layers.Embedding(600, 64))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              optimizer='adam', 
              metrics=['accuracy'])

# Model training
history = model.fit(x=train_padded, 
                    y=train_labels, 
                    epochs=20, 
                    batch_size=1,
                    validation_split=0.15, 
                    verbose=1,
                    callbacks=[checkpoint])

# Saving the model
model.save('sentiment_model_without_pretrained.h5')

model = load_model('best_model.h5')


# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_padded, test_labels)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Plotting the training & validation accuracy
plt.figure(figsize=(10,5))

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Epochs vs. Training and Validation Accuracy')
plt.legend()

# Plotting the training & validation loss values
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epochs vs. Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()