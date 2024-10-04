from indian_data import get_indian_data
from model_utils import mifs_selector_indian, kfold_division
from model import get_results


if __name__ == "__main__":
    print("__________________________________Indian Data______________________________________")
    path = "Indian dataSet/data/newDatafile.csv"
    data, label_encoder = get_indian_data(path)
    features_selected = mifs_selector_indian(data, k=12)
    data_selected = data[features_selected]
    data_kfold = kfold_division(data_selected, 5)
    y = data['WQC']
    X = data_selected.copy()
    get_results(X, y)
    # for num_folds in range(2, 11):
    #     print("__________________Number of Folds: {} __________________".format(num_folds))
    #     data_kfold = kfold_division(data_selected, num_folds=num_folds)
    #     y = data['WQC']
    #     X = data_selected.copy()
    #     get_results(X, y, num_folds=num_folds)
