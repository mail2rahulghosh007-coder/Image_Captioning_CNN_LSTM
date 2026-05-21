# Image Captioning using CNN and LSTM

## Overview
An image captioning system that automatically generates 
text descriptions for images using DenseNet + LSTM.

## Model Architecture
- **CNN**: DenseNet201 for image feature extraction
- **LSTM**: For caption sequence generation
- **Dataset**: Flickr8k (8000 images, 40000 captions)

## Results
![sample](input_imgs/sample.jpg)
> Caption: "young boy in blue shirt is playing in the water"

## Project Structure
├── input_imgs/       # Sample test images
├── models/           # Saved model files (download from Drive)
├── main.py           # Streamlit web app
├── requirements.txt  # Dependencies
└── image_captioning.ipynb  # Training notebook