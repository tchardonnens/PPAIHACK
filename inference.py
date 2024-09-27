import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader, Dataset


# Load the model architecture
with open('config.json', 'r') as file:
    config = json.load(file)

# Recreate the model architecture
class EEGClassifier(torch.nn.Module):
    def __init__(self):
        super(EEGClassifier, self).__init__()
        for layer in config['layers']:
            if layer['type'] == 'Conv2d':
                setattr(self, layer['name'], torch.nn.Conv2d(**layer['parameters']))
            elif layer['type'] == 'Linear':
                setattr(self, layer['name'], torch.nn.Linear(**layer['parameters']))
            elif layer['type'] == 'MaxPool2d':
                setattr(self, layer['name'], torch.nn.MaxPool2d(**layer['parameters']))

    def forward(self, x):
        for layer in config['layers']:
            if layer['type'] in ['Conv2d', 'Linear']:
                x = torch.relu(getattr(self, layer['name'])(x))
            elif layer['type'] == 'MaxPool2d':
                x = getattr(self, layer['name'])(x)
        return x
    
model = EEGClassifier()
model.load_state_dict(torch.load(config['model']['safetensors_path']))
model.to(config['inference']['device'])
model.eval()

# Prepare the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = load_dataset("JLB-JLB/seizure_eeg_dev")

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