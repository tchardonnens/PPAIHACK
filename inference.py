import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset


# Define the EEGNet model
class EEGNet(nn.Module):
    def __init__(self, num_classes=3):
        super(EEGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# Load configuration
with open('config.json', 'r') as file:
    config = json.load(file)

# Load the model
model = EEGNet()
model.load_state_dict(load_file(config['model']['safetensors_path']))
model.to(config['inference']['device'])
model.eval()

# Prepare the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Custom dataset class
class SeizureEEGDataset(Dataset):
    def __init__(self, hf_dataset, split='train', transform=None):
        self.data = hf_dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        if image.mode != 'L':
            image = image.convert('L')
        label = self.data[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
dataset = load_dataset("JLB-JLB/seizure_eeg_dev")
test_dataset = SeizureEEGDataset(dataset, split='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], 
                         shuffle=False, num_workers=config['data']['num_workers'])

# Perform inference
all_preds = []
all_labels = []

device = torch.device(config['inference']['device'])

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Print classification report
print(classification_report(all_labels, all_preds))

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calculate and print overall accuracy
accuracy = (cm.diagonal().sum() / cm.sum()) * 100
print(f"Overall Accuracy: {accuracy:.2f}%")

# Calculate and print per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
for i, acc in enumerate(class_accuracy):
    print(f"Accuracy of class {i}: {acc:.2f}%")
