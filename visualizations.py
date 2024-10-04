import numpy as np
import matplotlib.pyplot as plt
import seaborn
from utils import *
from preprocessing import *
from model_utils import *
from model import *


def _preprocessing_pipeline(data_path_):
    print("Importing Data...")
    frame = get_excel_data(data_path_)
    print("Data Import Done!!")
    print("___________________Exploring Started____________________________")
    print("Describing Data...")
    describe(frame)
    print("Describing Done!!")
    print("Checking Null Values..")
    check_null(frame)
    print("Checking Null Done!!")
    print("Checking Data info...")
    check_info(frame)
    print("Checking Info Done!!")
    print("___________________Exploring Done____________________________")
    print("___________________Processing Started____________________________")
    print("Missing Values Preprocessing initiated...")
    frame = preprocess_missing_data(frame)
    print("Done!!")
    # print("Transforming features initiated...")
    # frame = transform_features(frame)
    # print("Done!!")
    print("Creating Water Quality index...")
    frame = get_wqi(frame)
    print("Done!!")
    print("___________________Processing Done____________________________")
    print("Removing Correlated Columns...")
    highly_correlated_cols = ['mg2', 'na', 'tds105', 'co2_depend', 'hardness_permanent']
    frame = drop_cols(frame, cols=highly_correlated_cols)
    print("Done!!")
    print("Encoding Labels...")
    frame = encode_labels(frame, label_col='wqc')
    frame = convert_all_numerical(frame)
    print("Done!!")
    return frame


def remove_outlier(frame, upper_bound, lower_bound):
    frame = frame.drop(upper_bound[0])
    frame = frame.drop(lower_bound[0])
    return frame


# Univariate Analysis
def univariate_analysis(frame):
    unv_analysis_cols_ = frame.columns
    plot_univariate_analysis(frame, unv_analysis_cols_)


# Bivariate Analysis
def bivariate_analysis(frame):
    biv_analysis_cols_ = []
    for cols in frame.columns:
        for cols2 in frame.columns:
            biv_analysis_cols_.append((cols, cols2))
    # biv_analysis_cols_ = [('k', 'cl'), ('k', 'fe2'), ('no3', 'k'), ('hco3', 'co2_free'), ('ca2', 'fe2')]
    plot_bivariate_analysis(frame, biv_analysis_cols_)


# Outlier Analysis
def outlier_analysis(frame):
    outlier_analysis_cols_ = frame.columns
    plot_outliers(frame, outlier_analysis_cols_)


def _get_other_model_results(X, y, num_folds=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = random_forest()
    xgb = xgboost_model()
    lgbm = lightgbm_model()
    ailgbm = ai_lgbm_model(X_train, y_train)
    start_time = time.time()
    rf, acc_rf = model_performance(rf, X_train, y_train, X_test, y_test)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time for RF model: {}".format(elapsed_time))
    print("Random Forest Individual Performance: {}".format(acc_rf))
    start_time = time.time()
    rf, acc_xgb = model_performance(xgb, X_train, y_train, X_test, y_test)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time for XGB model: {}".format(elapsed_time))
    print("XGBoost Individual Performance: {}".format(acc_xgb))
    start_time = time.time()
    rf, acc_lgbm = model_performance(lgbm, X_train, y_train, X_test, y_test)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time for LightGBM model: {}".format(elapsed_time))
    print("LightGBM Individual Performance: {}".format(acc_lgbm))
    start_time = time.time()
    rf, acc_ailgbm = model_performance(ailgbm, X_train, y_train, X_test, y_test)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time for AILGBM model: {}".format(elapsed_time))
    print("AI-LightGBM Individual Performance: {}".format(acc_ailgbm))


def _get_results(X, y, num_folds=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # print(X_test)

    # Define models
    rf = random_forest()
    xgb = xgboost_model()
    lgbm = lightgbm_model()
    ailgbm = ai_lgbm_model(X_train, y_train)
    meta_model = meta_learning_model()
    models = [rf, xgb, lgbm, ailgbm]

    # Apply k-fold cross-validation
    results = kfold_cv(models, meta_model, X, y, k=num_folds)

    rf_pred = rf.predict_proba(X_test)
    xgb_pred = xgb.predict_proba(X_test)
    lgbm_pred = lgbm.predict_proba(X_test)
    ailgbm_pred = ailgbm.predict_proba(X_test)

    # Combine the predictions of the base models using the maximum probability as the choosing criterion
    ensemble_pred = np.maximum.reduce([rf_pred, xgb_pred, lgbm_pred, ailgbm_pred])
    ensemble_pred_class = np.argmax(ensemble_pred, axis=1)
    # print(ensemble_pred_class)

    def ensemble_predict(X_):
        # Predict probabilities from each base model
        rf_pred_ = rf.predict_proba(X_)[:, 1]
        xgb_pred_ = xgb.predict_proba(X_)[:, 1]
        lgbm_pred_ = lgbm.predict_proba(X_)[:, 1]
        ailgbm_pred_ = ailgbm.predict_proba(X_)[:, 1]
        # Take the maximum probability from the base models
        ensemble_pred_ = np.maximum.reduce([rf_pred_, xgb_pred_, lgbm_pred_, ailgbm_pred_])
        return np.vstack([1 - ensemble_pred_, ensemble_pred_]).T

    # Evaluate the ensemble model on the test set
    ensemble_acc = accuracy_score(y_test, ensemble_pred_class)
    print("Ensemble accuracy:", ensemble_acc)

    _get_other_model_results(X, y, num_folds=num_folds)
    #
    # feature_names = X_train.columns
    # target_names = ["wqc"]
    # # Create the LIME explainer
    # explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names,
    #                                                    class_names=target_names, discretize_continuous=False)
    #
    # # Choose a random instance from the test set
    # instance = X_test.values[10]
    #
    # # Explain the prediction of the ensemble model on the instance using LIME
    # exp = explainer.explain_instance(instance, ensemble_predict)
    # exp.save_to_file(f"./results/lime_explained.html")


# Number of Fold Comparison

def num_of_folds_analysis(frame):
    for num_folds in range(2, 11):
        print("__________________Number of Folds: {} __________________".format(num_folds))
        data_kfold = kfold_division(data_selected, num_folds=num_folds)
        y = data['WQC']
        X = data_selected.copy()
        _get_results(X, y, num_folds=num_folds)

# MIFS Score comparison

# Meta Learning Results Comparison

# Individual Model Comparison

# Runtime Comparison

# AI Optimization Information

# Explainable Models Analysis


if __name__ == "__main__":
    data_path = "./Vietnam dataset-phase-3/data-phase-3/daluong.xlsx"
    data = _preprocessing_pipeline(data_path)
    print(data.columns)
    # Univariate Analysis
    univariate_analysis(data)
    # Bivariate Analysis
    bivariate_analysis(data)
    # Outlier Analysis
    outlier_analysis(data)
    print("Done!!")
