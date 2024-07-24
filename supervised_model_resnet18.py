import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from tqdm import tqdm

# Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class SupervisedLearning:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.model = self._initialize_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.00001, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.5)  # Learning rate scheduler
        self.data_transforms = self._initialize_transforms()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []

    def _initialize_model(self):
         
        # Load ResNet-18 pretrained on ImageNet
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
        return model.to(device)    
    
    def _initialize_transforms(self):
        return {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def load_data(self):
        # Load the train, val, and test datasets
        train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.data_transforms['train'])
        val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.data_transforms['val'])
        test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=self.data_transforms['test'])

        # Create DataLoaders for each set
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def train_model(self, num_epochs=10):
        best_val_loss = float('inf')
        self.model.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for inputs, labels in self.train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)

                    pbar.update(1)

            train_loss = running_loss / len(self.train_loader)
            train_accuracy = correct_predictions / total_predictions
            val_loss = self.validate_model()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}, Train Accuracy: {train_accuracy}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model, 'latest_model.pth')  # Save the entire model

            self.scheduler.step()  # Update learning rate
        

    def validate_model(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        self.model.train()
        return val_loss / len(self.val_loader)

    def test_model(self):
        self.model = torch.load('latest_model.pth')
        self.model.eval()
        y_true = []
        y_scores = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                probabilities = nn.Softmax(dim=1)(outputs)
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probabilities[:, 1].cpu().numpy())
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        eer, eer_threshold = self.calculate_eer(y_true, y_scores)
        print(f"EER: {eer*100}, EER Threshold: {eer_threshold*100}")
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def calculate_eer(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return eer, eer_threshold     

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure()
        
        # Plot Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.train_losses, 'g', label='Training loss')
        plt.plot(epochs, self.val_losses, 'b', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.train_accuracies, 'g', label='Training accuracy')
        plt.title('Training accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    data_dir = r"E:\fasdataset_best_1"  # data directory path
    sl = SupervisedLearning(data_dir=data_dir, batch_size=32)
    sl.load_data()
    sl.train_model(num_epochs=20)
    sl.test_model()

if __name__ == "__main__":
    main()
