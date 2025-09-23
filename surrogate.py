import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# Step 1: Preprocessing function
# ------------------------------
def build_features_targets(data_dir, csv_path, save=True):
    """
    Build (features, targets) from Lorenz-96 simulations.

    Args:
        data_dir (str): Path to folder with X_*.npy files.
        csv_path (str): Path to targets_time_mean_energy.csv (contains F values).
        save (bool): Whether to save features.npy and targets.npy.
    
    Returns:
        features (np.ndarray): shape (N, 41)
        targets (np.ndarray): shape (N, 40)
        ids (list of str): list of run IDs
    """
    df = pd.read_csv(csv_path)

    features = []
    targets = []
    ids = []

    for _, row in df.iterrows():
        run_id = int(row['id'])
        F = row["F"]

        file_path = os.path.join(data_dir, f"X_{int(row['id']):07d}.npy")

        if not os.path.exists(file_path):
            print(f"File missing: {file_path}, skipping...")
            continue

        d = np.load(file_path)  # shape (40, 600)

        x0 = d[:, 0]     # first time step (40,)
        xT = d[:, -1]    # last time step (40,)

        features.append(np.concatenate([[F], x0]))  # (41,)
        targets.append(xT)                          # (40,)
        ids.append(run_id)

    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    if save:
        np.save(os.path.join(data_dir, "features.npy"), features)
        np.save(os.path.join(data_dir, "targets.npy"), targets)
        print(f"Saved: features.npy {features.shape}, targets.npy {targets.shape}")

    return features, targets, ids


# ------------------------------
# Step 2: PyTorch Dataset
# ------------------------------
class LorenzDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ------------------------------
# Step 3: Split + DataLoader
# ------------------------------
def get_dataloaders(features, targets, batch_size=16, test_size=0.15, val_size=0.15):
    ids = np.arange(len(features))

    # first split: train+val vs test
    trainval_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=42)
    # second split: train vs val
    train_ids, val_ids = train_test_split(trainval_ids, test_size=val_size/(1-test_size), random_state=42)

    X_train, y_train = features[train_ids], targets[train_ids]
    X_val, y_val = features[val_ids], targets[val_ids]
    X_test, y_test = features[test_ids], targets[test_ids]

    train_ds = LorenzDataset(X_train, y_train)
    val_ds = LorenzDataset(X_val, y_val)
    test_ds = LorenzDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class SurrogateNN(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=[64, 32], output_dim=40):
        super(SurrogateNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], output_dim)
        )

    def forward(self, x):
        return self.net(x)

def baseline_linear(features, targets):
    """Fit linear regression and evaluate on train/test split."""
    train_ids, test_ids = train_test_split(np.arange(len(features)), test_size=0.15, random_state=42)
    X_train, X_test = features[train_ids], features[test_ids]
    y_train, y_test = targets[train_ids], targets[test_ids]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression → RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return model

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        batch_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        # Validation
        model.eval()
        batch_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = criterion(preds, y)
                batch_losses.append(loss.item())
        val_loss = np.mean(batch_losses)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train RMSE: {np.sqrt(train_loss):.4f} "
              f"Val RMSE: {np.sqrt(val_loss):.4f}")

    return model, train_losses, val_losses

def evaluate_model(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred, multioutput="variance_weighted")
    print(f"Test RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return y_true, y_pred
