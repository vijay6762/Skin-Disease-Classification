import os
import io
import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from model import build_model

app = FastAPI(
    title="Skin Disease Prediction API",
    description="API for classifying dermatological images using PyTorch EfficientNet-B3"
)

# ---------------------------------------------------------
# SETUP: Load your trained model and classes dynamically
# ---------------------------------------------------------
MODEL_PATH = 'best_model.pth'
CLASSES_PATH = 'class_indices.json'
IMG_SIZE = (300, 300)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
CLASSES = []

# Load classes
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, 'r') as f:
        class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        CLASSES = [idx_to_class[i] for i in range(len(idx_to_class))]
        print(f"Loaded {len(CLASSES)} classes from mapping.")

# Load model
if os.path.exists(MODEL_PATH) and len(CLASSES) > 0:
    model = build_model(num_classes=len(CLASSES), fine_tune=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"PyTorch Model loaded successfully on {device}.")
else:
    print(f"Warning: {MODEL_PATH} not found or classes missing. Please train the model first.")

# Image preprocessing matching data_loader.py validation transforms
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    if model is None:
        return {"message": "API running, but PyTorch model not loaded (best_model.pth not found)."}
    return {"message": f"Skin Disease Prediction PyTorch API is running on {device}!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None or not CLASSES:
        return JSONResponse({"error": "Model or classes not loaded. Train the model first."}, status_code=500)
        
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()
        
        # Build dictionary of all class probabilities
        all_probs = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
        
        return JSONResponse({
            "status": "success",
            "predicted_class": CLASSES[predicted_class_idx],
            "confidence": confidence,
            "all_probabilities": all_probs
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)