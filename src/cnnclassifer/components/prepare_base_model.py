import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torch.nn as nn
import torchvision.models as models
from cnnclassifer.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        # Load a pre-trained VGG16 model in PyTorch
        self.model = models.vgg16(pretrained=(self.config.params_weights == "imagenet"))
        
        # Adjust the input size if needed (e.g., by modifying the first layer), but PyTorch doesn't have a specific input shape setting for models
        if not self.config.params_include_top:
            # Remove the final classification layer
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

        # Save the base model
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # Freeze layers
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for i, layer in enumerate(list(model.features) + list(model.classifier)):
                if i < freeze_till:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Identify the in_features of the last Linear layer
        in_features = None
        for layer in reversed(model.classifier):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                break

        # Ensure in_features was found before proceeding
        if in_features is None:
            raise ValueError("Could not find the last Linear layer in the model's classifier.")

        # Replace the last Linear layer with a new one for the specified number of classes
        model.classifier[-1] = nn.Linear(in_features, classes)

        # Define the loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        print(model)  # Print model summary (like Keras' `model.summary()`)

        return model, criterion, optimizer

    def update_base_model(self):
        self.full_model, self.criterion, self.optimizer = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the updated base model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model):
        # Save the model's state_dict for easier loading in PyTorch
        torch.save(model.state_dict(), path)
