from fastapi import APIRouter, UploadFile, File
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
from pathlib import Path
from torchvision import models

router = APIRouter(prefix="/face", tags=["Face Emotion Detection"])

class FERACK_EfficientNetB2(nn.Module):
    def __init__(self, num_classes=8, freeze_backbone=False):
        super().__init__()

        self.backbone = models.efficientnet_b2(weights=None)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
label_dataset = {
    0: 'Anger',
    1: 'Contempt',
    2: 'Disgust',
    3: 'Fear',
    4: 'Happy',
    5: 'Neutral',
    6: 'Sad',
    7: 'Surprise'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FERACK_EfficientNetB2()

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR.parent.parent / "model" / "Face Model" / "FERACK_EfficientNetB2_best_weights_B4.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

model.eval()

transform_val = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3, 1, 1)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    img = Image.open(file.file)
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()
    
    pred_result = {}

    for i in range(len(pred)):
        pred_result[label_dataset[i]] = round(float(pred[i]) * 100, 2)
    
    pred_result = dict(sorted(pred_result.items(), key= lambda x: x[1], reverse=True))

    return {"Emotions": pred_result}