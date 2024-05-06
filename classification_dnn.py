import numpy as np 
import pandas as pd
import os
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.models as models
from torch.nn.parallel import DataParallel
from torchvision.transforms import ToTensor
from torchvision import transforms
import torchvision.utils as vutils
import torch.optim as optim
from torchinfo import summary
from torchmetrics.image import StructuralSimilarityIndexMeasure

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PRETRAINED = False
SEED = 6
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.2
NUM_EPOCHS = 25

DATASET_PATH = '../Images/Train'
PRETRAINED_MODEL_PATH = '../dnn.pt'
TEST_PATH = '../Images/Test'
torch.manual_seed(SEED)

class ClassifierDataset(Dataset):
    '''Classifier Dataset object'''
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.image_folder = ImageFolder(root=self.root_dir)
        self.class_to_idx = self.image_folder.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.samples = self.image_folder.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._load_image(img_path)
        return image, label

    def _load_image(self, img_path):
        image = Image.open(img_path)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop((225,224)),
            transforms.Lambda(lambda img: img.crop((0, 0, img.width, img.height - 1))),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image_tensor = transform(image)
        return image_tensor
    
class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = self._load_image(img_path)
        return image

    def _load_image(self, img_path):
        image = Image.open(img_path)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop((225, 224)),
            transforms.Lambda(lambda img: img.crop((0, 0, img.width, img.height - 1))),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image_tensor = transform(image)
        return image_tensor , img_path.split('/')[-1][:-9]
    
rtx_dataset = TestDataset(root_dir=TEST_PATH)
rtx_dl = DataLoader(rtx_dataset, batch_size=1)
cnn_prediction = {}

custom_dataset = ClassifierDataset(root_dir=DATASET_PATH)
val_size = int(len(custom_dataset) * TRAIN_TEST_SPLIT)
train_size = len(custom_dataset) - val_size
train_ds, val_ds = random_split(custom_dataset, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = models.densenet121(weights=None)
model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_classes = 2
model.classifier = nn.Linear(model.classifier.in_features, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

device = torch.device("mps")
model = model.to(device)
model = DataParallel(model)

train_losses = []
val_losses = []
train_accu = []
val_accu = []

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_dl:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        train_loss += loss.item()

    train_accuracy = 100 * train_correct / train_total

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_dl:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_loss /= len(val_dl)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accu.append(train_accuracy)
    val_accu.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]: "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

model.eval()
with torch.no_grad():
    for inputs, label in rtx_dl:
        inputs = inputs.to(device)
        outputs = model(inputs)
        a, predicted = torch.max(outputs, 1)
        cnn_prediction[label[0]] = predicted.item()

for key, value in cnn_prediction.items():
    cnn_prediction[key] = 'Non Stochastic' if value == 0 else 'Stochastic'

data = {
    'Time Series': ['kai', 'gamma', 'phi', 'delta', 'mu', 'kappa', 'nu', 'theta', 'beta', 'lambda', 'rho', 'alpha'],
    'IISER': ['Stochastic', 'Stochastic', 'Stochastic', 'Non Stochastic', 'C', 'C', 'Non Stochastic', 'Non Stochastic', 'C', 'C', 'Non Stochastic', 'Non Stochastic'],
    'CI': ['Stochastic', 'Stochastic', 'Stochastic', 'Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic'],
    'DS': ['Stochastic', 'Stochastic', 'Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic']
}

df = pd.DataFrame(data)
df = pd.merge(df, pd.DataFrame(list(cnn_prediction.items()), columns=['Time Series', 'cnn_prediction']), on='Time Series', how='left')
print(df)