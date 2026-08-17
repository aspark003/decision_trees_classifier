import pandas as pd
from trees.decision_t import load_data, decision_tree, d_scores, d_evaluate, roc_auc, val_curve, pr_curve, lr_curve, calibration
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

    prediction_df, prediction_scores, figx_score = d_scores(dtc_model, X_train, y_train)

    eval_df, figx_bar, figx_tree = d_evaluate(dtc_model, X_train)

    roc, figx_curve = roc_auc(dtc_model, X_train, y_train)

    val_summary_df, figx_val = val_curve(dtc_model, X_train, y_train)

    avg_pre_recall, figx_pre = pr_curve(dtc_model, X_train, y_train)

    train_size, train_scores, test_scores, figx_lr = lr_curve(dtc_model, X_train, y_train)

    figx_cal = calibration(dtc_model, X_train, y_train)

   
    return dtc_model, X_train, X_test, y_train, y_test, prediction_df, prediction_scores, figx_score, eval_df, figx_bar, figx_tree, roc, figx_curve, val_summary_df, figx_val, avg_pre_recall, figx_pre, train_size, train_scores, test_scores, figx_lr, figx_cal    


if __name__ == "__main__":

    dtc_model, X_train, X_test, y_train, y_test, prediction_df, prediction_scores, figx_score, eval_df, figx_bar, figx_tree, roc, figx_curve, val_summary_df, figx_val, avg_pre_recall, figx_pre, train_size, train_scores, test_scores, figx_lr, figx_cal = dt_class()

    print(val_summary_df)

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed: {elapsed:.3f}")