import tkinter as tk
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model (replace 'your_model.h5' with the actual model file path)
model = load_model('E:\KANISHK\projects_null_class\English-to-Hindi Word Translator\s2s.h5')

# Load tokenizer if needed (replace 'tokenizer.pkl' with the actual tokenizer file path)
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess input text
def preprocess_input(text):
    # Tokenize and pad input text for the model (customize as per your model)
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100)  # assuming max length is 100
    return padded_sequences

# Function to perform prediction using the loaded model
def perform_prediction():
    input_text = input_entry.get("1.0", "end-1c")
    if input_text.strip() == "":
        messagebox.showerror("Input Error", "Please enter text to analyze")
        return
    
    # Preprocess the input
    preprocessed_text = preprocess_input(input_text)
    
    # Get the prediction
    prediction = model.predict(preprocessed_text)
    
    # Mock result interpretation (you can change this based on your model's output)
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    
    # Display the result
    output_entry.delete("1.0", "end")
    output_entry.insert("1.0", f"Prediction: {sentiment}")

# Initialize the main window
root = tk.Tk()
root.title("Text Sentiment Analyzer")
root.geometry("400x300")

# Input Text Label and Text Box
input_label = tk.Label(root, text="Enter Text:")
input_label.pack(pady=5)
input_entry = tk.Text(root, height=5, width=40)
input_entry.pack(pady=5)

# Predict Button
predict_button = tk.Button(root, text="Analyze", command=perform_prediction)
predict_button.pack(pady=10)

# Output Text Label and Text Box
output_label = tk.Label(root, text="Prediction Result:")
output_label.pack(pady=5)
output_entry = tk.Text(root, height=5, width=40)
output_entry.pack(pady=5)

# Run the main loop
root.mainloop()
