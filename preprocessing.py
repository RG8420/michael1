import numpy as np
from data import get_excel_data
from explore import describe, check_info, check_null
from sklearn import preprocessing


def impute(frame, cols):
    for col in cols:
        frame[col] = frame[col].fillna(value=frame[col].mean())
    return frame


def drop_cols(frame, cols):
    frame = frame.drop(cols, axis=1)
    return frame


def dropna_cols(frame, cols):
    frame = frame.dropna(subset=cols)
    return frame


def preprocess_missing_data(frame):
    drop_columns = ['number_analyzing', 'nh4', 'po4', 'eh', 'oxygen',
                    'lienhe', 'conductivity', 'oxygen_dissolve', 'tds180']
    frame = drop_cols(frame, drop_columns)
    na_cols_drop = ['laboratory', 'smell', 'tatse', 'color', 'quarter', 'well_code']
    frame = dropna_cols(frame, na_cols_drop)
    na_cols = ['fe2', 'fe3', 'no2', 'no3', 'co2_infiltrate', 'sio2']
    frame = impute(frame, na_cols)
    return frame


def transform_features(frame):
    frame['ph'] = frame['ph'].apply(lambda x: 1 if (5.5 <= x <= 8.5) else 0)
    frame['fe2'] = frame['fe2'].apply(lambda x: 1 if x <= 5 else 0)
    frame['fe3'] = frame['fe3'].apply(lambda x: 1 if x <= 5 else 0)
    frame['cl'] = frame['cl'].apply(lambda x: 1 if x <= 250 else 0)
    frame['so4'] = frame['so4'].apply(lambda x: 1 if x <= 400 else 0)
    frame['tds105'] = frame['tds105'].apply(lambda x: 1 if x <= 400 else 0)  # Total Dissolved Solids (TDS)
    frame['hardness_general'] = frame['hardness_general'].apply(
        lambda x: 1 if x <= 500 else 0)  # hardness_general is same as Total Hardness (TH)
    frame['no2'] = frame['no2'].apply(lambda x: 1 if x <= 1 else 0)
    frame['no3'] = frame['no3'].apply(lambda x: 1 if x <= 15 else 0)
    frame['ca2'] = frame['ca2'].apply(lambda x: 1 if x <= 75 else 0)
    frame['mg2'] = frame['mg2'].apply(lambda x: 1 if x <= 50 else 0)
    frame['na'] = frame['na'].apply(lambda x: 1 if x <= 200 else 0)
    frame['k'] = frame['k'].apply(lambda x: 1 if x <= 12 else 0)
    frame['hco3'] = frame['hco3'].apply(lambda x: 1 if x <= 350 else 0)
    frame['co3'] = frame['co3'].apply(lambda x: 1 if x <= 1 else 0)  # mean_value = 1
    frame['hardness_temporal'] = frame['hardness_temporal'].apply(lambda x: 1)
    frame['hardness_permanent'] = frame['hardness_permanent'].apply(lambda x: 1)
    frame['co2_free'] = frame['co2_free'].apply(lambda x: 1)
    frame['co2_depend'] = frame['co2_depend'].apply(lambda x: 1)
    frame['co2_infiltrate'] = frame['co2_infiltrate'].apply(lambda x: 1)
    frame['sio2'] = frame['sio2'].apply(lambda x: 1)
    frame['is_drinkable'] = frame.apply(lambda x: 1 if ((x["na"] == x["k"]) and (x["k"] == x["ca2"]) and
                                                        (x["ca2"] == x["mg2"]) and (x["mg2"] == x["fe3"]) and (
                                                                x["fe3"] == x["fe2"]) and (x["fe2"] == x["cl"])
                                                        and (x["cl"] == x["so4"]) and (x["so4"] == x["hco3"]) and (
                                                                x["hco3"] == x["co3"]) and (
                                                                x["co3"] == x["no2"]) and (
                                                                x["no2"] == x["no3"] and (x["no3"] == x["ph"])
                                                                and (x["ph"] == x["sio2"]) and (
                                                                        x["sio2"] == x["tds105"]) and (
                                                                        x["tds105"] == x["hardness_general"]) and (
                                                                        x["hardness_general"] == x["hardness_permanent"]
                                                                )
                                                                and (x["hardness_permanent"] == x["hardness_temporal"])
                                                                and (
                                                                        x["hardness_temporal"] == x["co2_free"]) and (
                                                                        x["co2_free"] == x["co2_depend"])
                                                                and (x["co2_depend"] == x["co2_infiltrate"]))
                                                        ) else 0, axis=1)
    return frame


def water_quality_class(x):
    if 0 <= x <= 25:
        return "Poor"
    elif 25 < x <= 50:
        return "Fair"
    elif 50 < x <= 70:
        return "Medium"
    elif 70 < x <= 90:
        return "Good"
    elif 90 < x <= 100:
        return "Excellent"


def encode_labels(frame, label_col):
    le = preprocessing.LabelEncoder()
    frame[label_col] = le.fit_transform(frame[label_col])
    return frame


def get_wqi(frame):
    #  standard value recommended for parameter
    si = 200 + 12 + 75 + 50 + 5 + 5 + 250 + 400 + 350 + 1 + 1 + 15 + 8.5 + 400 + 500 + 242 + 393 + 51 + 127 + 16 + 5
    k = 1 / si
    values = [200, 12, 75, 50, 5, 5, 250, 400, 350, 1, 1, 15, 8.5, 400, 500, 242, 393, 51, 127, 16, 5]
    wi = []
    for i in values:
        wi.append(k / i)

    ph_norm = []
    for i in frame['ph']:
        ph_norm.append(100 * ((i - 7.0) / (8.5 - 7.0)))

    fe2_norm = []
    for i in frame['fe2']:
        fe2_norm.append(100 * (i / 5))

    fe3_norm = []
    for i in frame['fe3']:
        fe3_norm.append(100 * (i / 5))

    so4_norm = []
    for i in frame['so4']:
        so4_norm.append(100 * (i / 400))

    tds105_norm = []
    for i in frame['tds105']:
        tds105_norm.append(100 * (i / 400))

    hardness_general_norm = []
    for i in frame['hardness_general']:
        hardness_general_norm.append(100 * (i / 500))

    no2_norm = []
    for i in frame['no2']:
        no2_norm.append(100 * (i / 1))

    no3_norm = []
    for i in frame['no3']:
        no3_norm.append(100 * (i / 15))

    ca2_norm = []
    for i in frame['ca2']:
        ca2_norm.append(100 * (i / 75))

    mg2_norm = []
    for i in frame['mg2']:
        mg2_norm.append(100 * (i / 50))

    co2_free_norm = []
    for i in frame['co2_free']:
        co2_free_norm.append(100 * (i / 51))

    co2_depend_norm = []
    for i in frame['co2_depend']:
        co2_depend_norm.append(100 * (i / 127))

    co2_infiltrate_norm = []
    for i in frame['co2_infiltrate']:
        co2_infiltrate_norm.append(100 * (i / 16))

    sio2_norm = []
    for i in frame['sio2']:
        sio2_norm.append(100 * (i / 42))

    na_norm = []
    for i in frame['na']:
        na_norm.append(100 * (i / 200))

    k_norm = []
    for i in frame['k']:
        k_norm.append(100 * (i / 12))

    hco3_norm = []
    for i in frame['hco3']:
        hco3_norm.append(100 * (i / 350))

    co3_norm = []
    for i in frame['co3']:
        co3_norm.append(100 * (i / 1))

    cl_norm = []
    for i in frame['cl']:
        cl_norm.append(100 * (i / 250))

    hardness_temporal_norm = []
    for i in frame['hardness_temporal']:
        hardness_temporal_norm.append(100 * (i / 242))

    hardness_permanent_norm = []
    for i in frame['hardness_permanent']:
        hardness_permanent_norm.append(100 * (i / 393))

    wqi = []
    for i in range(len(frame)):
        wqi.append(((ph_norm[i] * wi[0]) +
                    (fe2_norm[i] * wi[1]) +
                    (fe3_norm[i] * wi[2]) +
                    (so4_norm[i] * wi[3]) +
                    (tds105_norm[i] * wi[4]) +
                    (hardness_general_norm[i] * wi[5]) +
                    (no2_norm[i] * wi[6]) +
                    (no3_norm[i] * wi[7]) +
                    (ca2_norm[i] * wi[8]) +
                    (mg2_norm[i] * wi[9]) +
                    (co2_free_norm[i] * wi[10]) +
                    (co2_depend_norm[i] * wi[11]) +
                    (co2_infiltrate_norm[i] * wi[12]) +
                    (sio2_norm[i] * wi[13]) +
                    (na_norm[i] * wi[14]) +
                    (k_norm[i] * wi[15]) +
                    (hco3_norm[i] * wi[16]) +
                    (co3_norm[i] * wi[17]) +
                    (cl_norm[i] * wi[18]) +
                    (hardness_temporal_norm[i] * wi[19]) +
                    (hardness_permanent_norm[i] * wi[20])
                    ) / (wi[0] + wi[1] + wi[2] + wi[3] + wi[4] + wi[5] + wi[6] + wi[7] + wi[8] + wi[9] + wi[10] +
                         wi[11] + wi[12] + wi[13] + wi[14] + wi[15] + wi[16] + wi[17] + wi[18] + wi[19] + wi[20]))
    frame['wqi'] = wqi
    max_wqi = frame['wqi'].max()
    min_wqi = frame['wqi'].min()
    frame['wqi'] = frame['wqi'].apply(lambda x: ((x - min_wqi) / (max_wqi - min_wqi)) * 100)
    frame['wqc'] = frame['wqi'].apply(water_quality_class)
    return frame


def convert_all_numerical(frame):
    le = preprocessing.LabelEncoder()
    frame['well_code'] = le.fit_transform(frame['well_code'])
    frame['color'] = le.fit_transform(frame['color'])
    frame['smell'] = le.fit_transform(frame['smell'])
    frame['tatse'] = le.fit_transform(frame['tatse'])
    frame['quarter'] = le.fit_transform(frame['quarter'])
    frame['laboratory'] = le.fit_transform(frame['laboratory'])
    frame['date_sampling'] = (frame["date_sampling"] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
    frame['date_analyzing'] = (frame["date_analyzing"] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
    return frame


def preprocessing_pipeline(data_path):
    print("Importing Data...")
    frame = get_excel_data(data_path)
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
    print("Transforming features initiated...")
    frame = transform_features(frame)
    print("Done!!")
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


if __name__ == "__main__":
    path = "./Vietnam dataset-phase-3/data-phase-3/daluong.xlsx"
    print("Getting the Data...")
    data = get_excel_data(path)
    print("Done!!")
    print("Missing Values Preprocessing initiated...")
    data = preprocess_missing_data(data)
    print("Done!!")
    print("Transforming features initiated...")
    data = transform_features(data)
    print("Done!!")
    print("Creating Water Quality index...")
    data = get_wqi(data)
    data = convert_all_numerical(data)
    print(data.head())
    print("Done!!")
