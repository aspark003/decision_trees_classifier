import os
import time

import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, LearningCurveDisplay, ValidationCurveDisplay, validation_curve, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, PrecisionRecallDisplay, roc_auc_score, RocCurveDisplay, average_precision_score
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

start = time.time()

def load_data():
    """
    Download the heart-failure dataset and return it as a DataFrame
    """
    data = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    df = pd.read_csv(os.path.join(data, "heart.csv"))
    return df

def decision_tree(d):
    """
    Encode categorical columns, split the data, and train a decision tree.

    Args:
        d: The heart-failure DataFrame.

    Returns:
        The fitted model and the train_test_split
        (dtc_model, X_train, X_test, y_train, y_test).
    """

    cat_list = ['Sex','ChestPainType','RestingECG', 'ExerciseAngina','ST_Slope']
    d = pd.get_dummies(data=d, prefix=cat_list, columns=cat_list, dtype=int)
    features = [x for x in d.columns if x != 'HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(d[features], d['HeartDisease'], test_size=0.3, random_state=42)
    dtc_model = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4, max_features=None, min_impurity_decrease=0.01,
                                       min_samples_leaf=3, min_samples_split=2, min_weight_fraction_leaf=0.0, splitter='best', random_state=42)
    dtc_model.fit(X_train, y_train)

    return dtc_model, X_train, X_test, y_train, y_test

def d_scores(dtc_model, X_train, y_train):
    """
    Utilize dtc_model for prediction.
    
    Args:
        dtc_model: The fitted decision tree.
        X_train: The training features.
        y_train: The training labels.
    
    Returns:
        prediction_df: Per-row actual, predicted, and cross-val predicted labels, plus the overfit flag.
        prediction_scores: Overfit, cross-validation, and accuracy scores in one DataFrame.
        figx: The confusion matrix figure.
        (prediction_df, prediction_scores, figx)
    
    """
    d_pred = dtc_model.predict(X_train)
    d_cross_val_pred = cross_val_predict(dtc_model, X_train, y_train)
    prediction_df = pd.DataFrame({'Actual': y_train,
                                  'Predict': d_pred,
                                  'CV_predict': d_cross_val_pred}, index=X_train.index)
    prediction_df['Predict != CV_predict'] = (prediction_df['Predict'] != prediction_df['CV_predict']).astype(int)

    one = prediction_df[prediction_df['Predict != CV_predict'] == 1] 

    overfit = (len(one) / len(prediction_df)) * 100
    
    cv_score = (cross_val_score(dtc_model, X_train, y_train, cv=5).mean()) * 100

    accuracy = (dtc_model.score(X_train, y_train)) * 100
    
    prediction_scores = pd.DataFrame({'Overfit': overfit, 'Cross-val-score': cv_score, 'Accuracy-score': accuracy}, index=[0])

    class_report = classification_report(y_train, dtc_model.predict(X_train))

    confu_matrix = confusion_matrix(y_train, dtc_model.predict(X_train))

    figx_score, ax = plt.subplots(1,1, figsize=(10,8))
    ConfusionMatrixDisplay.from_estimator(dtc_model, X_train, y_train, ax=ax)
    figx_score.savefig("plots/confusion.png")

    return prediction_df, prediction_scores, class_report, confu_matrix, figx_score

def d_evaluate(dtc_model, X_train):
    """
    Collect data of feature importances.

    Args:
        dtc_model: Fitted decision tree.
        X_train: Training features
        
    Returns:
        eval_df: Feature importance DataFrame, order by feature importance.
        figx: Bar plot.
        figx1: Tree plot.
        (eval_df, figx, figx1)

    """

    feature_importance = dtc_model.feature_importances_

    eval_df = pd.DataFrame({'Feature-importance': feature_importance * 100}, index=X_train.columns).sort_values('Feature-importance', ascending=False)

    figx_bar, ax = plt.subplots(1,1, figsize=(10,8))
    sns.barplot(x=feature_importance, y=X_train.columns, ax=ax).tick_params(labelsize=8)
    figx_bar.savefig("plots/bar.png")

    figx_tree, ax = plt.subplots(1,1, figsize=(20,5))
    plot_tree(dtc_model, feature_names=X_train.columns, class_names=['healthy', 'disease'], filled=True, ax=ax)
    figx_tree.savefig("plots/tree.png")
    return eval_df, figx_bar, figx_tree

def roc_auc(dtc_model, X_train, y_train):
    """
    Generate roc_auc_score.

    Args:
        dtc_model: Fitted to get prediction probabilities.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        roc: roc-auc-scores.
        figx: Roc-Curve-Display.
        (roc, figx)

    """

    roc = roc_auc_score(y_train, dtc_model.predict_proba(X_train)[:,1]) * 100

    figx_curve, ax = plt.subplots(1,1, figsize=(10,8))
    RocCurveDisplay.from_estimator(dtc_model, X_train, y_train, ax=ax)
    figx_curve.savefig("plots/roc.png")
    return roc, figx_curve

def val_curve(dtc_model, X_train, y_train):
    """
    Generate Validation curve scores and plot.

    Args:
        dtc_model: Fitted to validation curve.
        X_train: Training Features.
        y_train: Training Labels.

    Returns:
        val_summary_df: Score for each fold; train / test.
        figx: Validation Curve Display.
        (val_summary_df, figx)    
    """

    train_score, test_score = validation_curve(dtc_model, X_train, y_train, param_name="max_depth", param_range=range(1,11))
    train_score = train_score.mean(axis=1)
    test_score = test_score.mean(axis=1)
    val_summary_df = pd.DataFrame({'Train fold': train_score, 'Test fold': test_score},index=range(1,11))
    val_summary_df = val_summary_df.sort_values('Test fold', ascending=False)

    figx_val, ax = plt.subplots(1,1, figsize=(10,8))
    ValidationCurveDisplay.from_estimator(dtc_model, X_train, y_train, param_name='max_depth', param_range=range(1,11), ax=ax)
    figx_val.savefig("plots/val_curve.png")
    return val_summary_df, figx_val

def pr_curve(dtc_model, X_train, y_train):
    """
    Generate Average precision recall.

    Args:
        dtc_model: Fitted to average-precision-score.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        avg_pre_recall: Average precision.
        figx: Precision Recall Display.
        (avg_pre_recall, figx)
    """
    avg_pre_recall = average_precision_score(y_train, dtc_model.predict_proba(X_train)[:,1]) * 100

    figx_pre, ax = plt.subplots(1,1, figsize=(10,8))
    PrecisionRecallDisplay.from_estimator(dtc_model, X_train, y_train, ax=ax)
    figx_pre.savefig("plots/precision_recall.png")
    return avg_pre_recall, figx_pre

def lr_curve(dtc_model, X_train, y_train):
    """
    Generate Learning curve.

    Args:
        dtc_model: Fitted to learning curve.
        X_train: Training Features.
        y_train: Training Labels.

    Returns:
        train_size: Sample counts per training size.
        train_scores: Training scores for each fold.
        test_scores: Test scores for each fold.
        figx: Learning Curve Display.
        (train_size, train_scores, test_scores, figx)
    """

    train_size, train_scores, test_scores = learning_curve(dtc_model, X_train, y_train, train_sizes=[0.1,0.3,0.6,0.8,1.0])

    figx_lr, ax = plt.subplots(1,1, figsize=(10,8))
    LearningCurveDisplay.from_estimator(dtc_model, X_train, y_train, ax=ax)
    figx_lr.savefig("plots/learning_curve.png")
    return train_size, train_scores, test_scores, figx_lr

def calibration(dtc_model, X_train, y_train):
    """
    Generate Calibration display.

    Args:
        dtc_model: Fitted to Calibration Display.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        figx: Calibration Display.
        (figx)
    """
    figx_cal, ax = plt.subplots(1,1, figsize=(10,8))
    CalibrationDisplay.from_estimator(dtc_model, X_train, y_train, ax=ax)
    figx_cal.savefig("plots/calibration.png")
    return figx_cal

def test_results(dtc_model, X_test, y_test):
    result_test_accuracy = dtc_model.score(X_test, y_test) * 100
    result_test_roc_auc = roc_auc_score(y_test, dtc_model.predict_proba(X_test)[:,1]) * 100
    result_confusion_matrix = confusion_matrix(y_test, dtc_model.predict(X_test))
    result_class_report = classification_report(y_test, dtc_model.predict(X_test))
    return result_test_accuracy, result_test_roc_auc, result_confusion_matrix, result_class_report


if __name__ == "__main__":
    
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

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed: {elapsed:.3f}")
