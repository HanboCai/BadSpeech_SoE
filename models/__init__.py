from .resnet import *
from .lstm import *
from .kwt import *

available_models = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'lstm',
    'kwt',
]

def create_model(model_name, num_classes, in_channels):
    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'lstm':
        model = lstm(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'kwt':
        model = KWT(input_res=[80,87],patch_res=[80,1], num_classes=num_classes, dim=64, depth=12, heads=1, mlp_dim=256, pre_norm=True,emb_dropout = 0.1)
    return model
