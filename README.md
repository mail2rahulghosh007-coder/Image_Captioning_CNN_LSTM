# Image Captioning using CNN and LSTM

## Overview
An image captioning system that automatically generates text descriptions for images using DenseNet + LSTM.

## Model Architecture
- **CNN**: DenseNet201 for image feature extraction
- **LSTM**: For caption sequence generation
- **Dataset**: Flickr8k (8000 images, 40000 captions)

## Download Model Files
Place all files in the `models/` folder before running the application:

| File | Download |
| :--- | :--- |
| `model.keras` | [Download](https://drive.google.com/file/d/120kvxeZoolkyjI7VxF_DaVysnXvDOqw/view?usp=drive_link) |
| `feature_extractor.keras` | [Download](https://drive.google.com/file/d/12j-YBzVgeAnJDLYwxu9rQ9ydUBWHO6vs/view?usp=drive_link) |
| `tokenizer.pkl` | [Download](https://drive.google.com/file/d/12UayNS-fptNEGBYIqaNvoVt6EI-UJzeY/view?usp=drive_link) |

## Results
![sample](input_imgs/sample.jpg)
> **Caption:** "young boy in blue shirt is playing in the water"

## Project Structure
```text
├── input_imgs/         # Sample test images
├── models/             # Saved model files (download from Drive)
├── main.py             # Streamlit web app
├── requirements.txt    # Dependencies
└── image_captioning.ipynb # Training notebook
 ```

## How To Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit web app
streamlit run main.py
```