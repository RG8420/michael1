from data import get_excel_data
from explore import describe, check_info, check_null
from preprocessing import preprocessing_pipeline, preprocess_missing_data, transform_features, get_wqi
from model_utils import mifs_selector, kfold_division
from model import get_results


def main_code(path):
    frame = preprocessing_pipeline(path)
    return frame


if __name__ == "__main__":
    data_path = "./Vietnam dataset-phase-3/data-phase-3/daluong.xlsx"
    data = main_code(data_path)
    features_selected = mifs_selector(data, k=20)
    data_selected = data[features_selected]
    data_kfold = kfold_division(data_selected, num_folds=5)
    y = data['wqc']
    X = data_selected.copy()
    for fold_number in range(2, 10):
        print("_____________________Number of Folds: {}____________________".format(fold_number))
        get_results(X, y, num_folds=fold_number)

    # data.to_csv("./Vietnam dataset-phase-3/data-phase-3/daluong_processed.csv")

