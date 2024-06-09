from chessml import script, config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

class NN(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        _, preds = torch.max(outputs, 1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrices and biases for real and imaginary parts
        self.real_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.imag_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.real_bias = nn.Parameter(torch.randn(out_features))
        self.imag_bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # Split input into real and imaginary parts
        real_part = x[:, :self.in_features]
        imag_part = x[:, self.in_features:]

        # Compute real and imaginary parts of the output
        real_output = F.linear(real_part, self.real_weight) - F.linear(imag_part, self.imag_weight)
        imag_output = F.linear(real_part, self.imag_weight) + F.linear(imag_part, self.real_weight)

        # Add biases
        real_output += self.real_bias
        imag_output += self.imag_bias

        # Concatenate real and imaginary parts of the output
        output = torch.cat((real_output, imag_output), dim=1)

        return output

class ToComplex(nn.Module):
    def forward(self, x):
        B, N = x.shape
        zeros = torch.zeros(B, N, device=x.device)
        output = torch.cat((x, zeros), dim=1)
        return output

class ToModulus(nn.Module):
    def forward(self, x):
        B, _ = x.shape
        N = x.shape[1] // 2
        real_part = x[:, :N]
        imag_part = x[:, N:]
        modulus = torch.sqrt(real_part**2 + imag_part**2)
        return modulus

class ToReal(nn.Module):
    def forward(self, x):
        B, _ = x.shape
        N = x.shape[1] // 2
        real_part = x[:, :N]
        return real_part

class PlainNN(NN):
    def __init__(self, hidden_layer_num: int, hidden_layer_size: int):
        super().__init__()
        hidden_layers = []
        for i in range(hidden_layer_num):
            in_features = 4 if i == 0 else hidden_layer_size
            hidden_layers.append(nn.Linear(in_features, hidden_layer_size))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output = nn.Linear(hidden_layer_size, 3)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output(x)
        return x

class ComplexNN(NN):
    def __init__(self, hidden_layer_num: int, hidden_layer_size: int):
        super().__init__()
        self.to_complex = ToComplex()

        complex_hidden_layer_size = hidden_layer_size // 2
        hidden_layers = []
        for i in range(hidden_layer_num):
            in_features = 4 if i == 0 else complex_hidden_layer_size
            hidden_layers.append(ComplexLinear(in_features, complex_hidden_layer_size))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(complex_hidden_layer_size, 3)
        self.to_modulus = ToModulus()


    def forward(self, x):
        x = self.to_complex(x)
        x = self.hidden_layers(x)
        x = self.to_modulus(x)
        x = self.output(x)
        return x

@script
def main(args):
  batch_size = 16

  # Load the Iris dataset
  iris = load_iris()
  X, y = iris.data, iris.target

  # Split the data into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Standardize the data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Convert to PyTorch tensors
  X_train = torch.tensor(X_train, dtype=torch.float32)
  X_test = torch.tensor(X_test, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.long)
  y_test = torch.tensor(y_test, dtype=torch.long)

  # Create PyTorch datasets and dataloaders
  train_dataset = TensorDataset(X_train, y_train)
  test_dataset = TensorDataset(X_test, y_test)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  # model = PlainNN(hidden_layer_num=5, hidden_layer_size=8)
  model = ComplexNN(hidden_layer_num=5, hidden_layer_size=8)

  # Initialize a trainer
  trainer = pl.Trainer(
    max_epochs=1000,
    accelerator=config.accelerator,
    devices=config.devices,
    logger=TensorBoardLogger(config.logs.tensorboard_path),
  )

  # Train the model
  trainer.fit(model, train_loader, test_loader)
