import pandas as pd
from trees.decision_t import load_data, decision_tree, d_scores, d_evaluate, roc_auc, val_curve, pr_curve, lr_curve, calibration, test_results
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.tree import plot_tree
import seaborn as sns
pd.set_option("display.max_columns", None)

start = time.time()


def dt_class():
    df = load_data()

    dtc_model, X_train, X_test, y_train, y_test= decision_tree(df)

    prediction_df, prediction_scores, class_report, confu_matrix, figx_score = d_scores(dtc_model, X_train, y_train)

    eval_df, figx_bar, figx_tree = d_evaluate(dtc_model, X_train)

    roc, figx_curve = roc_auc(dtc_model, X_train, y_train)

    val_summary_df, figx_val = val_curve(dtc_model, X_train, y_train)

    avg_pre_recall, figx_pre = pr_curve(dtc_model, X_train, y_train)

    train_size, train_scores, test_scores, figx_lr = lr_curve(dtc_model, X_train, y_train)

    figx_cal = calibration(dtc_model, X_train, y_train)

    result_test_accuracy, result_test_roc_auc, result_confusion_matrix, result_class_report = test_results(dtc_model, X_test, y_test)

   
    return dtc_model, X_train, X_test, y_train, y_test, prediction_df, prediction_scores, class_report, confu_matrix, figx_score, eval_df, figx_bar, figx_tree, roc, figx_curve, val_summary_df, figx_val, avg_pre_recall, figx_pre, train_size, train_scores, test_scores, figx_lr, figx_cal, result_test_accuracy, result_test_roc_auc, result_confusion_matrix, result_class_report       


if __name__ == "__main__":

    dtc_model, X_train, X_test, y_train, y_test, prediction_df, prediction_scores, class_report, confu_matrix, figx_score, eval_df, figx_bar, figx_tree, roc, figx_curve, val_summary_df, figx_val, avg_pre_recall, figx_pre, train_size, train_scores, test_scores, figx_lr, figx_cal, result_test_accuracy, result_test_roc_auc, result_confusion_matrix, result_class_report  = dt_class()

    print(f"Prediction scores: \n{prediction_scores.round(2)}\n")
    print(f"ROC-AUC: \n{roc:.2f}\n")
    print(f"Average precision: \n{avg_pre_recall:.2f}\n")
    print(f"Confusion matrix: \n{confu_matrix}\n")
    print(f"Classification report: \n{classification_report(y_train, dtc_model.predict(X_train))}\n")
    print(f"Feature importance: \n{eval_df.round(2)}\n")
    print(f"Model details: \n{dtc_model}\n")
    print(f"Validation summary: \n{val_summary_df.round(2)}\n")
    print(f"Train size: \n{train_size}\n")
    print(f"Train scores: \n{train_scores.mean(axis=1).round(2)}\n")
    print(f"Test scores: \n{test_scores.mean(axis=1).round(2)}\n")

    print(f"Test accuracy: \n{result_test_accuracy:.2f}\n")
    print(f"Test ROC-AUC: \n{result_test_roc_auc:.2f}\n")
    print(f"Test confusion matrix: \n{result_confusion_matrix}\n")
    print(f"Test classification report: \n{result_class_report}\n")

    

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed: {elapsed:.3f}")