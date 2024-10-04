from preprocessing import preprocessing_pipeline, remove_outlier
import numpy as np


def outlier_bound(frame, col):
    # finding the 1st quartile
    q1 = np.quantile(frame[col], 0.25)

    # finding the 3rd quartile
    q3 = np.quantile(frame[col], 0.75)

    med = np.median(frame[col])

    # finding the iqr region
    iqr = q3 - q1

    # finding upper and lower whiskers
    upper_bound = np.where(frame[col] >= q3 + (1.5 * iqr))
    lower_bound = np.where(frame[col] <= q1 - (1.5 * iqr))
    return upper_bound, lower_bound


def correlation_analysis(frame, cols):
    corr_matrix = frame.corr()
    corr_threshold = 0.6

    print("---------------CHECKING FOR CORRELATION------------")
    for c in cols:
        for r in cols:
            if r < c:
                if abs(corr_matrix[r][c]) > corr_threshold:
                    print(c + " : " + r + " : " + str(corr_matrix[r][c]))


if __name__ == "__main__":
    data_path = "./Vietnam dataset-phase-3/data-phase-3/daluong.xlsx"
    data = preprocessing_pipeline(data_path)
    print(data.columns)
    # ub, lb = outlier_bound(data, col='cl')
    # data = remove_outlier(data, ub, lb)
    corr_cols = ['na', 'k', 'ca2', 'mg2', 'fe3', 'fe2', 'cl', 'so4', 'hco3',
                 'co3', 'no2', 'hardness_general', 'no3', 'hardness_temporal',
                 'hardness_permanent', 'ph', 'co2_free', 'co2_depend', 'co2_infiltrate',
                 'sio2', 'tds105']
    correlation_analysis(data, corr_cols)

    print("Done!!")
