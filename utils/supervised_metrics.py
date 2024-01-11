from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             average_precision_score)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1_result = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1_result
        }
