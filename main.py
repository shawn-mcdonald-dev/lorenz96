import os
from surrogate import (
    build_features_targets, 
    get_dataloaders, 
    baseline_linear, 
    SurrogateNN, 
    train_model, 
    evaluate_model
)

if __name__ == "__main__":
    # Paths
    data_dir = "data/l96_N40_T20_S600_1000"
    csv_path = os.path.join(data_dir, "targets_time_mean_energy.csv")

    # Load dataset
    features, targets, ids = build_features_targets(data_dir, csv_path, save=True)

    # Baseline linear regression
    baseline_model = baseline_linear(features, targets)

    # Data loaders with scaling
    train_loader, val_loader, test_loader, feature_scaler, target_scaler = get_dataloaders(features, targets, batch_size=32)

    # Train NN surrogate
    model = SurrogateNN()
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, patience=20)

    # Evaluate NN
    y_true, y_pred = evaluate_model(model, test_loader, target_scaler)
