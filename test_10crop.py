import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def test(dataloader, model, args, viz, device):
    model.eval()
    all_video_scores = []

    with torch.no_grad():
        for i, (input, label) in enumerate(dataloader):
            input = input.to(device)  # Input shape: (1, 32, 1024)

            # Model forward
            score_abnormal, score_normal, _, _, _, _, logits, _, _, feat_magnitudes = model(inputs=input)

            # Get per-frame (per-segment) anomaly scores
            logits = torch.squeeze(logits, 1)  # Shape: (1, 32, 1) → (1, 32)
            logits = torch.mean(logits, 0)      # Temporal average: (32,) → single score
            video_score = logits.cpu().detach().numpy().mean()

            all_video_scores.append(video_score)

    # Load ground truth labels
    gt = np.load('list/gt-wrongway.npy')  # Update path if needed

    all_video_scores = np.array(all_video_scores)

    assert len(gt) == len(all_video_scores), f"GT length ({len(gt)}) and pred length ({len(all_video_scores)}) mismatch!"

    # Calculate ROC AUC
    fpr, tpr, threshold = roc_curve(gt, all_video_scores)
    np.save('fpr.npy', fpr)
    np.save('tpr.npy', tpr)
    rec_auc = auc(fpr, tpr)
    print('Test ROC AUC:', rec_auc)

    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(gt, all_video_scores)
    pr_auc = auc(recall, precision)
    print('Test PR AUC:', pr_auc)

    # Optional: plot on visdom if enabled
    if viz is not None:
        viz.plot_lines('Test_PR_AUC', pr_auc)
        viz.plot_lines('Test_ROC_AUC', rec_auc)
        viz.lines('scores', all_video_scores)
        viz.lines('roc', tpr, fpr)

    return rec_auc
