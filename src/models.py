"""
Model definitions for Customer Churn Intelligence Platform.
Defines 5 models: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, PyTorch NN.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


def get_sklearn_models() -> dict:
    """Return dictionary of sklearn-compatible models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=42, scale_pos_weight=2.6,  # ~ratio of majority/minority
            eval_metric="logloss"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }


class ChurnNet(nn.Module):
    """3-layer MLP for churn prediction."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class PyTorchChurnClassifier:
    """Sklearn-compatible wrapper around the PyTorch ChurnNet."""

    def __init__(self, input_dim: int, epochs: int = 100, lr: float = 0.001, batch_size: int = 64):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChurnNet(input_dim).to(self.device)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.model.train()
        X_t = torch.FloatTensor(np.array(X)).to(self.device)
        y_t = torch.FloatTensor(np.array(y)).to(self.device)

        # Class weights for imbalanced data
        pos_weight = (y_t == 0).sum() / (y_t == 1).sum()
        criterion = nn.BCELoss(weight=None)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                # Apply higher weight to positive class
                weights = torch.where(y_batch == 1, pos_weight, torch.tensor(1.0).to(self.device))
                loss = nn.functional.binary_cross_entropy(outputs, y_batch, weight=weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss)

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(np.array(X)).to(self.device)
        with torch.no_grad():
            probs = self.model(X_t).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {"input_dim": self.input_dim, "epochs": self.epochs,
                "lr": self.lr, "batch_size": self.batch_size}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self
