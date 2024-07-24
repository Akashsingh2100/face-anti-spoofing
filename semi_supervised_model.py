import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_curve
import numpy as np
import random
from tqdm import tqdm

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ProgressiveLearning:
    def __init__(self, data_dir, labeled_sample_size, batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.labeled_sample_size = labeled_sample_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mu = 0.7  # Confidence threshold for pseudo-labels
        self.k = 0.08  # Coefficient for sample rate
        self.model = self._initialize_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.00001, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.5)  # Learning rate scheduler
        self.data_transforms = self._initialize_transforms()

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
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        class_indices = {cls: [] for cls in range(len(train_dataset.classes))}
        labeled_indices = []

        # Populate the class_indices dictionary
        half_sample_size = self.labeled_sample_size // 2
        i, j = 0, 0
        for idx, (_, label) in enumerate(train_dataset):
            if label == 0 and i < half_sample_size:
                labeled_indices.append(idx)
                i += 1
            elif label == 1 and j < half_sample_size:
                labeled_indices.append(idx)
                j += 1

        all_indices = set(range(len(train_dataset)))
        unlabeled_indices = list(all_indices - set(labeled_indices))

        self.labeled_dataset = Subset(train_dataset, labeled_indices)
        self.unlabeled_dataset = Subset(train_dataset, unlabeled_indices)

        self.labeled_loader = DataLoader(self.labeled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.unlabeled_loader = DataLoader(self.unlabeled_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def train_model(self, dataloader, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    
                    pbar.update(1)

            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = correct_predictions / total_samples
            self.scheduler.step()  # Update learning rate

    def generate_pseudo_labels(self):
        self.model.eval()
        pseudo_labels = []
        confidence_scores = []
        all_indices = []

        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.unlabeled_loader):
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                probabilities = nn.Softmax(dim=1)(outputs)
                confidences, pseudo_label = torch.max(probabilities, 1)
                indices = np.arange(i * self.unlabeled_loader.batch_size, min((i + 1) * self.unlabeled_loader.batch_size, len(self.unlabeled_loader.dataset)))
                pseudo_labels.extend(pseudo_label.cpu().numpy())
                confidence_scores.extend(confidences.cpu().numpy())
                all_indices.extend(indices)
        
        selected_indices = [idx for idx, conf in zip(all_indices, confidence_scores) if conf > self.mu]

        
        return selected_indices, pseudo_labels

    def progressive_learning(self):
        Ct = 0  # Counter for the total number of selected pseudo-labels
        t = 0  # Iteration counter
        N = len(self.unlabeled_dataset) + len(self.labeled_dataset)

        # Initial training on the labeled dataset
        self.train_model(self.labeled_loader, num_epochs=25)

        while Ct <= N:
            t += 1
            print(f"Iteration: {t}")
            selected_indices, pseudo_labels = self.generate_pseudo_labels()
            if not selected_indices:
                print("No confident pseudo-labels found or not enough balance. Stopping training.")
                break

            pseudo_labeled_dataset = Subset(self.unlabeled_dataset, selected_indices)
            Dt = ConcatDataset([self.labeled_dataset, pseudo_labeled_dataset])
            
            train_loader = DataLoader(Dt, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.train_model(train_loader, num_epochs=10)
            sigma = self.k * t
            Ct = int(sigma * N)
            

    def evaluate(self):
        self.model.eval()
        y_true = []
        y_scores = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                probabilities = nn.Softmax(dim=1)(outputs)
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probabilities[:, 1].cpu().numpy())  # Assuming class 1 is the positive class

        
        eer, eer_threshold = self.calculate_eer(y_true, y_scores)
        print(f"EER: {eer}, EER Threshold: {eer_threshold}")

    def calculate_eer(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return eer, eer_threshold

def main():
    data_dir = r"E:\fasdataset"  # Data directory path
    labeled_sample_size = 100  # Example size
    pl = ProgressiveLearning(data_dir, labeled_sample_size, batch_size=32)
    pl.load_data()
    pl.progressive_learning()
    pl.evaluate()

if __name__ == "__main__":
    main()
