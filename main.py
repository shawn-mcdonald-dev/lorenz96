import os
from surrogate import build_features_targets, get_dataloaders, baseline_linear, SurrogateNN, train_model, evaluate_model

if __name__ == "__main__":
  # Next steps: add arg parser here for data_dir, model params, training params, etc.
  # For now, hardcode paths and params
  data_dir = "data/l96_N40_T20_S600_1000"
  csv_path = os.path.join(data_dir, "targets_time_mean_energy.csv")

  # Load dataset
  features, targets, ids = build_features_targets(data_dir, csv_path, save=True)
  train_loader, val_loader, test_loader = get_dataloaders(features, targets, batch_size=16)

  # Baseline linear regression
  baseline_model = baseline_linear(features, targets)

  # Train NN
  model = SurrogateNN()
  model, train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)

  # Evaluate NN
  y_true, y_pred = evaluate_model(model, test_loader)
