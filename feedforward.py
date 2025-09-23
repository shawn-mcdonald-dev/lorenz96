import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from l96 import FlexibleNN, train_model, evaluate_model, plot_loss_curves, plot_predictions_vs_true

class L96Dataset(Dataset):
    def __init__(self, csv_file, data_dir, ids, scaler=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['id'].isin(ids)]
        self.data_dir = data_dir
        self.scaler = scaler

        self.inputs, self.targets = [], []
        for _, row in self.df.iterrows():
            id_str = row['id']
            F = row['F']
            target = row['E_time_mean']
            # load x0 vector
            x0_path = os.path.join(data_dir, f"x0_{int(id_str):07d}.npy")
            x0 = np.load(x0_path)
            x = np.concatenate([[F], x0])  # shape (41,)
            self.inputs.append(x)
            self.targets.append(target)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32).reshape(-1, 1)

        # Scale inputs if scaler provided
        if self.scaler:
            self.inputs = self.scaler.transform(self.inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

def feed_forward_model(data_dir, csv_file):
  # Read all IDs
  df = pd.read_csv(csv_file)
  ids = df['id'].tolist()

  # Train/val/test split
  train_ids, test_ids = train_test_split(ids, test_size=0.15, random_state=42)
  train_ids, val_ids = train_test_split(train_ids, test_size=0.176, random_state=42)  # 0.176*0.85 ≈ 0.15

  # Fit scaler on training inputs
  all_inputs = []
  for id_str in train_ids:
      row = df[df['id'] == id_str].iloc[0]
      F = row['F']
      x0 = np.load(os.path.join(data_dir, f"x0_{int(id_str):07d}.npy"))
      all_inputs.append(np.concatenate([[F], x0]))
  scaler = StandardScaler().fit(np.array(all_inputs, dtype=np.float32))

  # Create datasets
  train_ds = L96Dataset(csv_file, data_dir, train_ids, scaler)
  val_ds = L96Dataset(csv_file, data_dir, val_ids, scaler)
  test_ds = L96Dataset(csv_file, data_dir, test_ids, scaler)

  train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
  test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

  model = FlexibleNN(input_dim=41, hidden_layers=[64, 32])
  trained_model, history = train_model(model, train_loader, val_loader, epochs=50)
  train_losses = history["train_loss"]
  val_losses = history["val_loss"]
  
  plot_loss_curves(train_losses, val_losses)

  # Evaluate on test set
  preds, trues = evaluate_model(model, test_loader)

  # Example: compute RMSE
  rmse = np.sqrt(np.mean((preds - trues) ** 2))
  print(f"Test RMSE: {rmse:.6f}")

  # Add baseline here (linear model)

  plot_predictions_vs_true(trues, preds)