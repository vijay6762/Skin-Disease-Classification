import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes, fine_tune=False, fine_tune_layers=50):
        super().__init__()
        
        # Load EfficientNet-B3 with ImageNet weights
        # We use IMAGENET1K_V1 which is the standard strong baseline
        self.base_model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Features extraction part (the convolutional backbone)
        self.features = self.base_model.features
        
        # Replace the classifier head
        # EfficientNet-B3 feature extractor outputs 1536 channels
        in_features = 1536 
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self._set_fine_tuning(fine_tune, fine_tune_layers)

    def _set_fine_tuning(self, fine_tune, fine_tune_layers):
        # First, freeze all base model parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        if fine_tune:
            # Revert to list of modules to unfreeze top N layers
            feature_modules = list(self.features.children())
            
            # If fine_tune_layers is very large, just unfreeze everything
            if fine_tune_layers >= len(feature_modules):
                for param in self.features.parameters():
                    param.requires_grad = True
            else:
                # Unfreeze only the last N blocks
                for module in feature_modules[-fine_tune_layers:]:
                    for param in module.parameters():
                        param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

def build_model(num_classes, fine_tune=False, fine_tune_layers=50):
    return SkinDiseaseModel(num_classes, fine_tune, fine_tune_layers)

if __name__ == "__main__":
    model = build_model(num_classes=31)
    print("Model created successfully.")
