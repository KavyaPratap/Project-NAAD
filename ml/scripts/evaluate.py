# Owner: S2
# Module: Model Evaluation — Confusion Matrix + FAR
# Input:  ml/data/test_X.npy, ml/data/test_y.npy, ml/models/naad_kws_float.h5
# Output: docs/test_results/kws_eval_float.json
# Last validated: [date, commit]

import numpy as np
import tensorflow as tf
import json
import os

os.makedirs("docs/test_results", exist_ok=True)

def evaluate_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Full evaluation: accuracy, confusion matrix, FAR estimate."""
    model = tf.keras.models.load_model(model_path)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Confusion matrix
    TP = int(np.sum((y_pred == 1) & (y_test == 1)))
    FP = int(np.sum((y_pred == 1) & (y_test == 0)))
    TN = int(np.sum((y_pred == 0) & (y_test == 0)))
    FN = int(np.sum((y_pred == 0) & (y_test == 1)))

    recall     = TP / (TP + FN + 1e-9)
    precision  = TP / (TP + FP + 1e-9)
    f1         = 2 * (precision * recall) / (precision + recall + 1e-9)
    accuracy   = (TP + TN) / len(y_test)

    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Test samples: {len(y_test)}")
    print(f"  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    print(f"Recall (TPR): {recall:.4f} ({recall*100:.1f}%)")
    print(f"Precision:    {precision:.4f} ({precision*100:.1f}%)")
    print(f"F1 Score:     {f1:.4f}")
    print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"FAR estimate: {FP} false triggers in test set")
    print("=" * 50)

    return dict(recall=recall, precision=precision, f1=f1, accuracy=accuracy,
                TP=TP, FP=FP, TN=TN, FN=FN)


if __name__ == "__main__":
    X_test = np.load("ml/data/test_X.npy")
    y_test = np.load("ml/data/test_y.npy")

    print("\n--- Float model ---")
    r_float = evaluate_model("ml/models/naad_kws_float.h5", X_test, y_test)

    # Save results
    with open("docs/test_results/kws_eval_float.json", "w") as f:
        json.dump(r_float, f, indent=2)
    print("Results saved to docs/test_results/kws_eval_float.json")
