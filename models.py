import torch.nn as nn
from torchvision import models
import timm

def build_densenet121(num_classes: int = 4):
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

def build_vit(model_name: str, num_classes: int = 4):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    return model
