import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torch.nn as nn
import torchvision.models as models
import time
from cnnclassifer.entity.config_entity import TrainingConfig


import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

class Training:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_base_model()
        # Load the base model from the specified path
        
    def get_base_model(self):
        # Load a VGG16 model without pretrained weights
        self.model = models.vgg16(pretrained=(self.config.params_weights == "imagenet"))
        
        # Modify the final layer of the classifier to match the number of classes
        in_features = self.model.classifier[-1].in_features  # Get the input features of the last layer
        self.model.classifier[-1] = nn.Linear(in_features, self.config.params_classes)  # New output layer

        # Load the state_dict if available, with strict=False to handle any structural differences
        if Path(self.config.updated_base_model_path).exists():
            self.model.load_state_dict(torch.load(self.config.updated_base_model_path), strict=False)

        # Move the model to the specified device (GPU or CPU)
        self.model = self.model.to(self.device)

    def train_valid_generator(self):
        # Define transformations for data augmentation and normalization
        train_transform = transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.config.params_image_size[0]),
            transforms.RandomAffine(degrees=0, shear=0.2),
            transforms.RandomResizedCrop(self.config.params_image_size[0], scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) if self.config.params_is_augmentation else transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load training and validation data
        self.train_dataset = datasets.ImageFolder(root=self.config.training_data, transform=train_transform)
        self.valid_dataset = datasets.ImageFolder(root=self.config.training_data, transform=valid_transform)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    def train(self):
        # Set up the training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate)
        num_epochs = self.config.params_epochs

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for images, labels in self.valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == labels).item()

            val_loss /= len(self.valid_loader.dataset)
            val_accuracy = correct / len(self.valid_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Save the trained model
        self.save_model(path=self.config.trained_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model):
        torch.save(model.state_dict(), path)
