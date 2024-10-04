import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
# from main import main_code


def mifs_selector(df, k):
    # Calculate mutual information between each feature and target variable
    mi = mutual_info_classif(df.drop(columns=['wqc']), df['wqc'])
    # Sort features by mutual information in descending order
    sorted_features = sorted(list(zip(df.columns[:-1], mi)), key=lambda x: x[1], reverse=True)
    selected_features = [x[0] for x in sorted_features[:k]]
    return selected_features


def mifs_selector_indian(df, k):
    # Calculate mutual information between each feature and target variable
    mi = mutual_info_classif(df.drop(columns=['WQC']), df['WQC'])
    # Sort features by mutual information in descending order
    sorted_features = sorted(list(zip(df.columns[:-1], mi)), key=lambda x: x[1], reverse=True)
    print(sorted_features)
    selected_features = [x[0] for x in sorted_features[:k]]
    return selected_features


def kfold_division(df, num_folds):
    # Define the number of splits
    num_splits = 5

    # Define the K-fold cross-validator
    kfold = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    folds = kfold.split(df)
    return folds


if __name__ == "__main__":
    data_path = "./Vietnam dataset-phase-3/data-phase-3/daluong.xlsx"
    data = main_code(data_path)
    features_selected = mifs_selector(data, k=20)
    data_selected = data[features_selected]
    data_kfold = kfold_division(data_selected, num_folds=5)

