import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# 1. Dataset loader
# ---------------------------
def load_dataset(data_dir):
    features, targets = [], []

    for json_file in sorted(glob.glob(os.path.join(data_dir, "inputs_*.json"))):
        with open(json_file, "r") as f:
            info = json.load(f)

        F_value = info["F"]
        X_path = os.path.join(data_dir, info["outputs"]["X_path"])

        d = np.load(X_path)

        x0 = d[:, 0]   # (40,)
        xT = d[:, -1]  # (40,)

        feat = np.hstack([x0, F_value])  # (41,)
        features.append(feat)
        targets.append(xT)

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

# ---------------------------
# 2. Model definition
# ---------------------------
class SurrogateNetold(nn.Module):
    def __init__(self, input_dim=41, hidden=[64, 32], output_dim=40):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class SurrogateNet(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=[32, 16], output_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim[1], output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# 3. Training loop
# ---------------------------
def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> Saved new best model")
    
    return train_losses, val_losses

# ---------------------------
# Visualization Functions
# ---------------------------

def get_run_dir(base_dir="runs"):
    """Create a new folder for each run with timestamp."""
    os.makedirs(base_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def plot_loss_curves(train_losses, val_losses, run_dir):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300)


def plot_predictions(model, loader, device, y_mean, y_std, run_dir, n_samples=5):
    """Plot true vs predicted outputs for a few samples from the loader."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds) * y_std + y_mean
    trues = np.vstack(trues) * y_std + y_mean

    # Random subset
    idxs = np.random.choice(len(preds), size=min(n_samples, len(preds)), replace=False)
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(idxs, 1):
        plt.subplot(1, n_samples, i)
        plt.plot(trues[idx], label="True", marker="o")
        plt.plot(preds[idx], label="Pred", marker="x")
        plt.title(f"Sample {idx}")
        if i == 1:
            plt.legend()
    plt.tight_layout()
    save_path = os.path.join(run_dir, "predictions.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

# ---------------------------
# 4. Main
# ---------------------------
def main(args):
    # Load dataset
    X, y = load_dataset(args.data_dir)
    # print("X, y: ", X.shape, y.shape)

    # Normalize (optional, improves training)
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    y_mean, y_std = y.mean(0), y.std(0) + 1e-8
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    # Train/Val/Test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_norm, y_norm, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    # Build loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurrogateNet().to(device)

    # Train
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    # Test evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            test_loss += criterion(pred, yb).item() * xb.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Final Test Loss: {test_loss:.4f}")

    get_runs = get_run_dir()
    plot_loss_curves(train_losses, val_losses, get_runs)
    plot_predictions(model, test_loader, device, y_mean, y_std, get_runs, n_samples=3)
    print(f"Saved predictions plot to {get_runs}")
    # Save normalization stats
    # np.savez("norm_stats.npz", X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with X_*.npy files")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
