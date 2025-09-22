import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ========================
# Flexible Neural Network
# ========================
class FlexibleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32]):
        super(FlexibleNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # scalar output
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ========================
# Training Pipeline
# ========================
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item() * x.size(0)
        
        val_loss /= len(val_loader.dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return model, history

# ========================
# Evaluation
# ========================
def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds.append(model(X).cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return preds, trues

# ========================
# Visualization
# ========================
def plot_loss_curves(train_losses, val_losses):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss [64, 32]")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")
    plt.show()

def plot_predictions_vs_true(y_true, y_pred):
    """Scatter plot: predicted vs. true values."""
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
    plt.xlabel("True E_time_mean")
    plt.ylabel("Predicted E_time_mean")
    plt.title("Predicted vs. True")
    plt.legend()
    plt.grid(True)
    plt.savefig("preds_vs_true.png")
    plt.show()

def plot_val_vs_rmse(results, rmses):
    archs = list(results.keys())
    val_losses = [results[a] for a in archs]
    test_rmses = [rmses[a] for a in archs]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Architecture")
    ax1.set_ylabel("Validation Loss", color="tab:blue")
    ax1.plot(archs, val_losses, marker="o", color="tab:blue", label="Validation Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Test RMSE", color="tab:red")
    ax2.plot(archs, test_rmses, marker="s", color="tab:red", label="Test RMSE")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Validation Loss vs Test RMSE by Architecture")
    fig.tight_layout()
    plt.show()
