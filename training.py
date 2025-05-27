import torch
import torch.nn as nn
import torch.optim as optim

def evaluate(model, loader, verbose=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            output = model(x_batch)  # (batch, T, num_classes)
            preds = output.argmax(dim=-1)
            all_preds.append(preds.flatten())
            all_labels.append(y_batch.flatten())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).float().mean().item()
    if verbose:
        print(f"accuracy: {acc:.4f}")
    return acc

def train(model, loader, optimizer, criterion, val_loader=None, epochs=100):
    model.train()
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(x_batch)  # shape: (batch, T, num_classes)
            loss = criterion(output.view(-1, output.shape[-1]), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)

        # Validation accuracy
        if val_loader:
            acc = evaluate(model, val_loader, verbose=False)
            val_accuracies.append(acc)

        print(f"epoch {epoch+1}: loss = {avg_loss:.4f}" + (f", val. acc. = {acc:.4f}" if val_loader else ""))

    return train_losses, val_accuracies
