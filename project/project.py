import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None, is_train=True):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train
        self.target_columns = ['X4_mean', 'X11_mean',
                               'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]  # Assuming the first column is 'id'
        img_name = f"{self.image_dir}/{img_id}.jpeg"
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        ancillary_data = self.data.iloc[idx, 1:].drop(
            self.target_columns, errors='ignore').values
        if self.is_train:
            targets = self.data.loc[idx, self.target_columns].values
            return image, torch.tensor(ancillary_data, dtype=torch.float64), torch.tensor(targets, dtype=torch.float64), img_id
        else:
            return image, torch.tensor(ancillary_data, dtype=torch.float64), img_id


# Data Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Train Dataset and DataLoader
train_dataset = CustomDataset(image_dir='data/train_images',
                              csv_file='data/train.csv', transform=transform, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Test Dataset and DataLoader
test_dataset = CustomDataset(image_dir='data/test_images',
                             csv_file='data/test.csv', transform=transform, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Model Definition
class CombinedModel(nn.Module):
    def __init__(self, num_ancillary_features, num_targets):
        super(CombinedModel, self).__init__()
        self.vgg16 = models.vgg16(weights='DEFAULT')
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-3])
        self.fc1 = nn.Linear(num_ancillary_features, 128, dtype=torch.float64)
        self.fc2 = nn.Linear(128, 128, dtype=torch.float64)
        self.fc3 = nn.Linear(128 + 4096, 128, dtype=torch.float64)
        self.fc4 = nn.Linear(128, num_targets, dtype=torch.float64)

    def forward(self, image, ancillary_data):
        x = self.vgg16(image)
        y = torch.relu(self.fc1(ancillary_data))
        y = torch.relu(self.fc2(y))
        combined = torch.cat((x, y), dim=1)
        z = torch.relu(self.fc3(combined))
        output = self.fc4(z)
        return output


# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_ancillary_features = len(train_dataset[0][1])
num_targets = len(train_dataset.target_columns)

model = CombinedModel(num_ancillary_features, num_targets).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0., 0.9))

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, ancillary_data, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, ancillary_data, targets = images.to(
            device), ancillary_data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images, ancillary_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Create Submission File
def create_submission(model, test_loader, submission_file='submission.csv'):
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for images, ancillary_data, ids_batch in test_loader:
            images, ancillary_data = images.to(
                device), ancillary_data.to(device)
            outputs = model(images, ancillary_data)
            predictions.append(outputs.cpu().numpy())
            ids.extend(ids_batch)

    predictions = np.concatenate(predictions, axis=0)

    with open(submission_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'X4', 'X11', 'X18', 'X26', 'X50', 'X3112'])
        for idx, pred in zip(ids, predictions):
            writer.writerow([idx] + list(pred))


# Create submission
create_submission(model, test_loader)
