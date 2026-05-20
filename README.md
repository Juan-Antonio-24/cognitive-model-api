# Cognitive Model API - Multimodal Emotion Detection

AI-powered multimodal emotion detection system using:

- Facial emotion recognition with Deep Learning (EfficientNet-B2)
- Text emotion classification using Transformers (RoBERTuito)
- FastAPI backend
- React frontend integration
- REST API architecture

---

# Features

- Facial emotion detection from images
- Text emotion analysis in Spanish
- Multimodal AI architecture
- FastAPI REST API
- Swagger documentation
- React frontend integration
- Transformer-based NLP
- CNN-based Computer Vision
- Modular backend architecture

---

# Technologies

## Backend
- Python
- FastAPI
- Uvicorn
- Pydantic

## Deep Learning
- PyTorch
- Torchvision
- Transformers (Hugging Face)

## NLP
- RoBERTuito
- Tokenizers

## Computer Vision
- EfficientNet-B2
- PIL
- Torchvision Transforms

## Frontend
- React

---

# Project Architecture

```text
Cognitive-model/
│
├── api/
│   ├── main.py
│   └── routers/
│       ├── face.py
│       └── text.py
│
├── model/
│   ├── Face Model/
│   └── Text Model/
│
├── training/
│   ├── Face Training/
│   └── Text Training/
│
├── requirements.txt
└── README.md
```

---

# Facial Emotion Detection

The facial emotion recognition model was developed using:

- EfficientNet-B2
- Transfer Learning
- Fine-Tuning
- PyTorch

## Dataset Sources

- FER2013
- AffectNet
- CK+

## Detected Emotions

- Anger
- Contempt
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

# Text Emotion Detection

The NLP emotion classifier was developed using:

- RoBERTuito
- Hugging Face Transformers
- PyTorch

## Language
- Spanish

## Detected Emotions

- joyful
- mad
- peaceful
- powerful
- sad
- scared

---

# API Endpoints

## Facial Emotion Detection

```http
POST /face/predict
```

Receives:
- Image file

Returns:
- Emotion probabilities

---

## Text Emotion Detection

```http
POST /text/predict
```

Receives:

```json
{
  "text": "Estoy muy feliz hoy"
}
```

Returns:
- Emotion probabilities

---

# Swagger Documentation

Available at:

```text
http://127.0.0.1:8000/docs
```

---

# Installation

## Clone repository

```bash
git clone https://github.com/your-username/cognitive-model.git
cd cognitive-model
```

---

## Create virtual environment

```bash
python -m venv .venv
```

---

## Activate virtual environment

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the API

```bash
uvicorn api.main:app --reload
```

---

# Frontend Integration

The backend supports integration with React applications through CORS middleware.

Example frontend deployment:

```text
https://ai-educational-frontend.vercel.app
```

---

# Important Notes

Some trained model weights are not included in this repository due to GitHub file size limitations.

Large files such as:

```text
model.safetensors
```

are excluded from version control.

The repository still includes:
- API architecture
- Training notebooks
- Model configurations
- Tokenizers
- Backend implementation
- Project structure

---

# Training Notebooks

## Face Model
- Emotion_Detection_FER2013_AffectNet_CK.ipynb

## Text Model
- Emotion_Detection_RoBERTuito.ipynb

---

# Future Improvements

- Real-time webcam emotion detection
- Audio emotion analysis
- Docker deployment
- CI/CD pipeline
- Model optimization and quantization
- Cloud deployment
- Authentication system
- Emotion analytics dashboard

---

# Author

Juan Mendes

Systems Engineering Student focused on:
- Artificial Intelligence
- Deep Learning
- NLP
- Computer Vision
- Multimodal AI Systems

---

# License

This project is for educational and portfolio purposes.
