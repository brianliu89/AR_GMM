import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.models.resnet import resnet50
import torchvision.models as models
from PIL import Image
import numpy as np

batch_size = 20

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root = 'data/', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root = 'data/', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size, shuffle=False
)

device = torch.device('cuda:0')
model = models.resnet18(pretrained=False).to(device)
model.load_state_dict(torch.load('./CIFAR10_Scartch_1_1.pth'), strict=False)
model.eval()

features = []
labels = []
selected_classes = [2, 3, 6]

def hook_function(module, input, output):
    global_avg_pool_output = F.adaptive_avg_pool2d(output, (1, 1))
    flat_output = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)
    features.append(flat_output.cpu().data)

layer = model.layer4
hook = layer.register_forward_hook(hook_function)

for images, targets in test_loader:
    masks = (targets == selected_classes[0]) | (targets == selected_classes[1]) | (targets == selected_classes[2])
    selected_images = images[masks]
    if len(selected_images) > 0:
        selected_images = selected_images.to(device)
        output = model(selected_images)

hook.remove()

features_np = np.vstack([feature_tensor.numpy() for feature_tensor in features])
np.save('features.npy', features_np) 