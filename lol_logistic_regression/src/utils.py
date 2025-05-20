import torch

def accuracy(outputs, labels):
    preds = (outputs >= 0.5).long()
    labels = labels.long()

    preds = preds.view(-1)
    labels = labels.view(-1)

    print("Preds shape:", preds.shape, "Labels shape:", labels.shape)
    print("Unique preds:", torch.unique(preds))
    print("Unique labels:", torch.unique(labels))

    correct = (preds == labels).sum().item()
    total = labels.size(0)
    print(f"Correct: {correct}, Total: {total}, Accuracy: {correct / total}")
    return correct / total
