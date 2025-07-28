import torch
import numpy as np
from model import Model
from dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)

def evaluate_model(model_path, gt_path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(n_features=args.feature_size, batch_size=args.batch_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False)

    all_scores = []

    with torch.no_grad():
        for input, label in test_loader:
            input = input.to(device)
            _, _, _, _, _, _, logits, _, _, _ = model(inputs=input)
            score = torch.squeeze(logits, 1).cpu().numpy().flatten().mean()
            all_scores.append(score)

    gt = np.load(gt_path)
    preds = np.array(all_scores)
    def find_best_f1_threshold(gt, preds):
        best_f1 = 0
        best_threshold = 0.5
        for t in np.linspace(0.01, 0.99, 100):
            binary_preds = (preds >= t).astype(int)
            f1 = f1_score(gt, binary_preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        return best_threshold, best_f1


    print("\n=== Metrics Evaluation ===")
    print(f"Scores shape: {preds.shape}, GT shape: {gt.shape}")
    assert len(preds) == len(gt), "Mismatch between predictions and ground truth!"

    # AUC Metrics
    fpr, tpr, _ = roc_curve(gt, preds)
    rec_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(gt, preds)
    pr_auc = auc(recall, precision)

    # Search best threshold
    best_thresh, best_f1 = find_best_f1_threshold(gt, preds)
    print(f"\n🔍 Best Threshold based on F1: {best_thresh:.2f} → F1 = {best_f1:.4f}")

    # Now apply it
    preds_binary = (preds >= best_thresh).astype(int)
    acc = accuracy_score(gt, preds_binary)
    f1 = f1_score(gt, preds_binary)
    cm = confusion_matrix(gt, preds_binary)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(gt, preds_binary))


# Usage Example
if __name__ == "__main__":
    import option
    args = option.parser.parse_args()

    evaluate_model(
        model_path='ckpt/rtfmfinal.pkl',
        gt_path='list/gt-wrongway.npy',
        args=args
    )
