# ASL Alphabet Recognition

This project is a web application for recognizing American Sign Language (ASL) alphabet signs using a deep learning model. The application is built with Streamlit and uses a pre-trained MobileNetV2 model from TensorFlow.

## Table of Contents

- Installation
- Usage
- Model
- Contributing

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/madeekon/AMLFinal_ASLGesture.git
   cd ASL-Alphabet-Recognition

   ```

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:
   pip install -r requirements.txt

## Usage

To start the Streamlit app, run the following command:

streamlit run app.py

Upload an image of an ASL alphabet sign, and the app will display the predicted class of the sign along with confidence scores.

## Model

The model used in this project is a MobileNetV2 pretrained on ImageNet and fine-tuned for ASL alphabet classification. The model is saved in Keras format and loaded within the Streamlit app for real-time predictions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
