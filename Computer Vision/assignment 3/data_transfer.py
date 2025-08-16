import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import torchmetrics

from torchvision import transforms
from torchvision.datasets import Imagenette
from torchvision.datasets import CIFAR10
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

class TransferCNN(L.LightningModule):
    def __init__(self, num_classes=10):
        super(TransferCNN, self).__init__()

        self.estimator = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling (GAP)  
        )

        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(p=0.3)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        x = self.estimator(x)
        x = self.dropout(x)
        x = self.classifier(x)  
        x = x.view(x.size(0), -1)
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.accuracy(y_hat, y)
        self.log("val_accuracy", self.accuracy)
        self.log("val_loss", loss)
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.accuracy(y_hat, y)
        self.log("test_accuracy", self.accuracy)
        self.log("test_loss", loss)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    # Define Data Transforms for CIFAR-10
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='data/cifar10', train=True, download=True, transform=train_transforms)
    test_dataset = CIFAR10(root='data/cifar10', train=False, download=True, transform=test_transforms)
    
    # Use 10% of the training set for validation
    train_set_size = int(len(train_dataset) * 0.9)
    val_set_size = len(train_dataset) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, val_set_size], generator=seed)
    
    # Load the pre-trained model from Imagenette
    checkpoint_path = "/content/lightning_logs/version_0/checkpoints/epoch=23-step=1608.ckpt" # Replace with the path of checkpoint if needed
    cifar_model = TransferCNN.load_from_checkpoint(checkpoint_path, num_classes=10)

    # Use DataLoader to load the dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False)
    
    # Add EarlyStopping
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=5)

    # Configure Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min"
    )

    # Fit the pre-trained model
    trainer = L.Trainer(callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model=cifar_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluate the model on the test set
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=8, shuffle=False)
    trainer.test(model=cifar_model, dataloaders=test_loader)