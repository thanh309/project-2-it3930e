import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for plotting functions

def predict_with_uncertainty(model, x, runs=50, device='cpu'):
    model.eval()
    preds = []

    with torch.no_grad():
        for _ in range(runs):
            output = model(x.to(device))  # shape: (1, T, num_classes)
            prob = F.softmax(output, dim=-1)  # shape: (1, T, num_classes)
            preds.append(prob.cpu().numpy())

    preds = np.stack(preds, axis=0)  # shape: (runs, batch, T, num_classes)
    mean_pred = preds.mean(axis=0)   # shape: (batch, T, num_classes)
    std_pred = preds.std(axis=0)     # shape: (batch, T, num_classes)

    return mean_pred, std_pred

def plot_prediction_with_uncertainty(mean_pred, std_pred, timestep=0, class_names=None):
    if class_names is None:
        class_names = [f'Class {i}' for i in range(mean_pred.shape[1])]

    probs = mean_pred[timestep]  # shape: (num_classes,)
    stds = std_pred[timestep]    # shape: (num_classes,)

    plt.figure(figsize=(5, 3))
    plt.bar(range(len(probs)), probs, yerr=stds, capsize=5, color='skyblue', alpha=0.8)
    plt.xticks(range(len(probs)), class_names)
    plt.ylabel("Predicted Probability")
    plt.title(f"Uncertainty at timestep {timestep}")
    plt.grid(axis='y')
    plt.show()

def bayesian_predictive_entropy(model, x, runs=10, device='cpu'):
    model.eval()
    predictions = []

    with torch.no_grad():
        for _ in range(runs):
            out = model(x.to(device))  # shape: (1, T, C)
            probs = F.softmax(out, dim=-1)  # shape: (1, T, C)
            predictions.append(probs.cpu().numpy())

    # shape: (runs, batch, T, C)
    preds = np.stack(predictions, axis=0)

    # average softmax predictions across runs
    mean_pred = preds.mean(axis=0)  # shape: (batch, T, C)

    # Entropy: H[p] = -sum p log p
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-12), axis=-1)  # shape: (batch, T)

    return mean_pred, entropy

def plot_entropy(entropy, title="Predictive uncertainty"):
    plt.figure(figsize=(6, 4))
    plt.plot(entropy, marker='o', color='purple')
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
