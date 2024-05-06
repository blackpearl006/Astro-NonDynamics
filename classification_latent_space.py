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
SEED = 45
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.2
NUM_EPOCHS = 50
EMBEDDING_CHANNELS = 8

OPTIMIZER = 'SGD'

DATASET_PATH = '../Images/Train'
PRETRAINED_MODEL_PATH = '../autoencoder.pt'
TEST_PATH = '../Images/Test'

def show(data1, data2):
    '''function to compare the reconstruction obtained by the autoencoder,
    SSMI tries to maintain the structural features
    MLE tries to lower the pixelwise differences
    '''
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))

    for i in range(1, 11):
        image_tensor1 = data1[i - 1]
        image_array1 = image_tensor1.numpy()
        image_tensor2 = data2[i - 1]
        image_array2 = image_tensor2.numpy()

        plt.subplot(2, 10, 2 * i - 1)
        plt.imshow(image_array1[0], cmap='gray')  # Assuming the first channel is the single channel
        plt.axis('off')

        plt.subplot(2, 10, 2 * i)
        plt.imshow(image_array2[0], cmap='gray')  # Assuming the first channel is the single channel
        plt.axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig('reconstructed.png')
    plt.close()

class AutoEncoderDataset(Dataset):
    '''AutoEncoder Dataset object, converts the images into single channel 
    '''
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

torch.manual_seed(SEED)
custom_dataset = AutoEncoderDataset(root_dir=DATASET_PATH)

class Autoencoder(nn.Module):
    def __init__(self, emb_size):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Flatten(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(1024*7*7, emb_size),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(emb_size, 1024*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (1024, 7, 7)),  # Reshape linear output to 2D shape

            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        bottleneck_output = self.bottleneck(x)
        x = self.decoder(bottleneck_output)
        return x, bottleneck_output
    
autoencoder_model = Autoencoder(EMBEDDING_CHANNELS)
summary(autoencoder_model, input_size=(BATCH_SIZE,1,224,224))

custom_dataset = AutoEncoderDataset(root_dir=DATASET_PATH)
val_size = int(len(custom_dataset) * TRAIN_TEST_SPLIT)
train_size = len(custom_dataset) - val_size
train_ds, val_ds = random_split(custom_dataset, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

if PRETRAINED :
    autoencoder_model = Autoencoder(EMBEDDING_CHANNELS)
    device = torch.device("mps")
    autoencoder_model = autoencoder_model.to(device)
    state_dict = torch.load(PRETRAINED_MODEL_PATH)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    autoencoder_model.load_state_dict(new_state_dict)
    autoencoder_model = DataParallel(autoencoder_model)
    criterion = nn.MSELoss()
#     criterion = StructuralSimilarityIndexMeasure(data_range=1.0)
    if OPTIMIZER == 'SGD':
        optimizer = optim.SGD(autoencoder_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    elif OPTIMIZER == 'Adam':
        optimizer = optim.Adam(autoencoder_model.parameters(), lr=LEARNING_RATE)
else :
    # #%%%%%%%%%%%%%%%%%%%%%%%     MODEL   %%%%%%%%%%%%%%%%%%%%%%%
    autoencoder_model = Autoencoder(EMBEDDING_CHANNELS)
    device = torch.device("mps")
    autoencoder_model = autoencoder_model.to(device)
    autoencoder_model = DataParallel(autoencoder_model)
    # criterion = nn.MSELoss()
    criterion = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    if OPTIMIZER == 'SGD':
        optimizer = optim.SGD(autoencoder_model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'Adam':
        optimizer = optim.Adam(autoencoder_model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        autoencoder_model.train()
        train_loss = 0.0
        for inputs, labels in train_dl:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = autoencoder_model(inputs)
            loss = 1 - criterion(outputs[0], inputs)
            # loss = criterion(outputs[0], inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            criterion.update(outputs[0], inputs)

        autoencoder_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs = inputs.to(device)
                outputs = autoencoder_model(inputs)
                loss = 1 - criterion(outputs[0], inputs)
                # loss = criterion(outputs[0], inputs)
                val_loss += loss.item()
                criterion.update(outputs[0], inputs)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]: Train Loss: {train_loss:.5f} Val Loss: {val_loss:.5f}")

    torch.save(autoencoder_model.state_dict(), 'autoencoder.pt')

autoencoder_model.eval()

with torch.no_grad():
    for inputs, labels in val_dl:
        inputs = inputs.to(device)
        outputs = autoencoder_model(inputs)
        show(inputs.to('cpu'), outputs[0].to('cpu'))
        break

embedded_data = {0:[], 1:[]}
embedding_dataset = AutoEncoderDataset(root_dir=DATASET_PATH)
embedding_dl = DataLoader(embedding_dataset, batch_size=BATCH_SIZE)

autoencoder_model.eval()
with torch.no_grad():
    for inputs, labels in embedding_dl:
        inputs = inputs.to(device)
        _, embedding = autoencoder_model(inputs)
        for k, class_ in enumerate(labels):
            embedded_data[class_.item()].append(embedding[k].cpu().numpy().reshape(1,-1)[0])

class_0_data = np.array(embedded_data[0])
class_1_data = np.array(embedded_data[1])

X = np.vstack((class_0_data, class_1_data))
y = np.hstack((np.zeros(class_0_data.shape[0]), np.ones(class_1_data.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logistic_reg_model = LogisticRegression()
logistic_reg_model.fit(X_train, y_train)

y_pred = logistic_reg_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on Test Set: {accuracy}')

class TestAutoEncoderDataset(Dataset):
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
    
rtx_dataset = TestAutoEncoderDataset(root_dir=TEST_PATH)
rtx_dl = DataLoader(rtx_dataset, batch_size=1)

rtx_embeddings = {}
autoencoder_model.eval()
with torch.no_grad():
    for inputs, label in rtx_dl:
        inputs = inputs.to(device)
        _, embedding = autoencoder_model(inputs)
        rtx_embeddings[label[0]] = embedding.cpu().squeeze().numpy().reshape(1,-1)[0]

rtx_names = np.array(list(rtx_embeddings.keys()))
rtx_X = np.array(list(rtx_embeddings.values()))
rtx_predictions = logistic_reg_model.predict(rtx_X)

data = {
    'Time Series': ['kai', 'gamma', 'phi', 'delta', 'mu', 'kappa', 'nu', 'theta', 'beta', 'lambda', 'rho', 'alpha'],
    'IISER': ['Stochastic', 'Stochastic', 'Stochastic', 'Non Stochastic', 'C', 'C', 'Non Stochastic', 'Non Stochastic', 'C', 'C', 'Non Stochastic', 'Non Stochastic'],
    'CI': ['Stochastic', 'Stochastic', 'Stochastic', 'Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic'],
    'DS': ['Stochastic', 'Stochastic', 'Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic', 'Non Stochastic']
}

df = pd.DataFrame(data)
data = []
for rtx_data, prediction in zip(rtx_names, rtx_predictions):
    predicted_label = 'Stochastic' if prediction == 1 else 'Non Stochastic'

    row = {
        'Time Series': rtx_data,
        'LogisticReg': predicted_label,
    }
    data.append(row)
    
df_new_predictions = pd.DataFrame(data)
df = pd.merge(df, df_new_predictions, on='Time Series', how='left')
print(df)

