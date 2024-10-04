# from preprocessing import preprocessing_pipeline
import time
import numpy as np
from data import get_csv_data
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import lime
import lime.lime_tabular
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# Define the affinity function
def affinity(X_train, y_train, params):
    # Create LightGBM classifier with specified hyperparameters
    # print(params)
    params = {key: value for key, value in params.items() if key in ['max_depth', 'learning_rate', 'num_leaves']}

    clf = lgb.LGBMClassifier(**params)

    # Train and evaluate model on training set
    # print(y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)

    # Compute affinity as inverse of classification error
    return 1 / (1 + accuracy)


# Define the immune optimization algorithm
def immune_optimization(X_train, y_train, param_ranges, num_ants=10, num_generations=100, mutation_rate=0.1):
    # Initialize population of antibodies
    num_params = len(param_ranges)
    population = [{} for i in range(num_ants)]
    for i in range(num_ants):
        for p in param_ranges.keys():
            population[i][p] = np.random.choice(param_ranges[p])

    # Iterate over generations
    best_params = None
    best_affinity = 0
    for gen in range(num_generations):
        # Compute affinity of each antibody
        affinities = [affinity(X_train, y_train, population[i]) for i in range(num_ants)]
        if max(affinities) > best_affinity:
            best_params = population[np.argmax(affinities)]
            best_affinity = max(affinities)

        # Select antibodies for cloning based on affinity
        clone_probs = affinities / np.sum(affinities)
        num_clones = int(np.floor(num_ants / 2))
        clones = []
        for i in range(num_clones):
            parent_idx = np.random.choice(range(num_ants), p=clone_probs)
            clones.append(population[parent_idx])

        # Mutate clones by randomly selecting a parameter and perturbing its value
        for clone in clones:
            for p, p_ in zip(param_ranges.keys(), range(num_params)):
                if np.random.uniform() < mutation_rate:
                    clone[p_] = np.random.choice(param_ranges[p])

        # Replace least-affine antibodies with clones
        replace_indices = np.argsort(affinities)[:num_clones]
        for i in range(num_clones):
            population[replace_indices[i]] = clones[i]
        # print(best_params)
        # print(best_affinity)

    return best_params, best_affinity


def random_forest():
    rf = RandomForestClassifier(n_estimators=100)
    return rf


def xgboost_model():
    xgb = XGBClassifier(n_estimators=100)
    return xgb


def lightgbm_model():
    lgbm = lgb.LGBMClassifier(n_estimators=100)
    return lgbm


def ai_lgbm_model(X_train, y_train):
    # Define the hyperparameter ranges to be tuned
    param_ranges = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [10, 20, 30]
    }

    # Run the Immune Optimization Algorithm
    best_params, best_affinity = immune_optimization(X_train, y_train, param_ranges)
    best_params = {key: value for key, value in best_params.items() if
                   key in ['max_depth', 'learning_rate', 'num_leaves']}
    print(best_params)
    print(best_affinity)

    # Create and evaluate the LightGBM classifier with the best hyperparameters
    clf = lgb.LGBMClassifier(**best_params)
    return clf


def meta_learning_model():
    meta_model = lgb.LGBMClassifier(n_estimators=100)
    return meta_model


# Define function for getting base model predictions
def get_base_predictions(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def model_performance(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc


# Define function for getting meta model predictions
def get_meta_predictions(meta_model, base_predictions):
    # dtrain = lgb.Dataset(base_predictions, label=y_train)
    # Find the index of the maximum value in each row
    # print(base_predictions.shape)
    max_index = np.argmax(base_predictions, axis=1)

    # Reshape the max_index array to shape (100, 1)
    y = np.reshape(max_index, (-1, 1)).astype("int")
    X_train, X_test, y_train, y_test = train_test_split(base_predictions, y, test_size=0.15, random_state=42)
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)
    print("Accuracy of Meta model: {}".format(accuracy_score(y_test, y_pred)))
    return meta_model, meta_model.predict(base_predictions)


# Define k-fold cross-validation function
def kfold_cv(models, meta_model, X, y, k=5):
    n = X.shape[0]
    fold_size = n // k
    results = np.zeros(n)
    all_base_predictions = None
    models_name = ["RF", "XGB", "LGBM", "AILGBM"]
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)
        X_test = X[start:end]
        idx = 0
        for j, model in enumerate(models):
            # print("Model Name: {}, Fold Number: {}".format(models_name[idx], i))
            # start_time = time.time()
            base_predictions = get_base_predictions(model, X_train, y_train, X_test)
            if i == 0:
                all_base_predictions = np.zeros((n, len(models)))
            all_base_predictions[start:end, j] = base_predictions
            # end_time = time.time()
            # Calculate elapsed time
            # elapsed_time = end_time - start_time
            # print("Elapsed time: ", elapsed_time)
            idx += 1
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)
        fold_base_predictions = all_base_predictions[start:end, :]
        # start_time = time.time()
        meta_model, meta_predictions = get_meta_predictions(meta_model, fold_base_predictions)
        # end_time = time.time()
        # print(meta_predictions)
        # Calculate elapsed time
        # elapsed_time = end_time - start_time
        # print("Elapsed time for Meta Prediction: {} for fold {}: ".format(elapsed_time, i))
        results[start:end] = meta_predictions.reshape(-1, )
    return results


def get_other_model_results(X, y, num_folds=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = random_forest()
    xgb = xgboost_model()
    lgbm = lightgbm_model()
    ailgbm = ai_lgbm_model(X_train, y_train)
    # start_time = time.time()
    rf, acc_rf = model_performance(rf, X_train, y_train, X_test, y_test)
    # end_time = time.time()
    # Calculate elapsed time
    # elapsed_time = end_time - start_time
    # print("Elapsed time for RF model: {}".format(elapsed_time))
    print("Random Forest Individual Performance: {}".format(acc_rf))
    # start_time = time.time()
    rf, acc_xgb = model_performance(xgb, X_train, y_train, X_test, y_test)
    # end_time = time.time()
    # Calculate elapsed time
    # elapsed_time = end_time - start_time
    # print("Elapsed time for XGB model: {}".format(elapsed_time))
    print("XGBoost Individual Performance: {}".format(acc_xgb))
    # start_time = time.time()
    rf, acc_lgbm = model_performance(lgbm, X_train, y_train, X_test, y_test)
    # end_time = time.time()
    # Calculate elapsed time
    # elapsed_time = end_time - start_time
    # print("Elapsed time for LightGBM model: {}".format(elapsed_time))
    print("LightGBM Individual Performance: {}".format(acc_lgbm))
    # start_time = time.time()
    rf, acc_ailgbm = model_performance(ailgbm, X_train, y_train, X_test, y_test)
    # end_time = time.time()
    # # Calculate elapsed time
    # elapsed_time = end_time - start_time
    # print("Elapsed time for AILGBM model: {}".format(elapsed_time))
    print("AI-LightGBM Individual Performance: {}".format(acc_ailgbm))


def get_results(X, y, num_folds=5):
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

    get_other_model_results(X, y, num_folds=num_folds)

    feature_names = X_train.columns
    target_names = ["wqc"]
    # Create the LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names,
                                                       class_names=target_names, discretize_continuous=False)

    # Choose a random instance from the test set
    index = 10
    instance = X_test.values[index]

    # Explain the prediction of the ensemble model on the instance using LIME
    exp = explainer.explain_instance(instance, ensemble_predict)
    exp.save_to_file(f"./results/lime_explained.html")
