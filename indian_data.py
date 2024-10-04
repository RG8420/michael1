import pandas as pd

from data import get_csv_data
from preprocessing import *
from sklearn.preprocessing import LabelEncoder


def preprocess_indian_missing_data(frame):
    return frame


def transform_indian_features(frame):
    frame2 = pd.read_csv("Indian dataSet/data/water_quality.csv")
    frame["Carbonate"] = frame2["Carbonate"]
    frame['pH'] = frame['pH'].apply(lambda x: 1 if (5.5 <= x <= 8.5) else 0)
    # frame['fe2'] = frame['fe2'].apply(lambda x: 1 if x <= 5 else 0)
    # frame['fe3'] = frame['fe3'].apply(lambda x: 1 if x <= 5 else 0)
    frame['Chloride'] = frame['Chloride'].apply(lambda x: 1 if x <= 250 else 0)
    frame['Sulphate'] = frame['Sulphate'].apply(lambda x: 1 if x <= 400 else 0)
    frame['TDS'] = frame['TDS'].apply(lambda x: 1 if x <= 400 else 0)  # Total Dissolved Solids (TDS)
    # frame['hardness_general'] = frame['hardness_general'].apply(
    #     lambda x: 1 if x <= 500 else 0)  # hardness_general is same as Total Hardness (TH)
    # frame['TDS'] = frame['TDS'].apply(lambda x: 1 if x <= 400 else 0)
    frame['EC'] = frame['EC'].apply(lambda x: 1 if x <= 750 else 0)
    frame['TH'] = frame['TH'].apply(lambda x: 1 if x <= 150 else 0)
    # frame['no2'] = frame['no2'].apply(lambda x: 1 if x <= 1 else 0)
    # frame['no3'] = frame['no3'].apply(lambda x: 1 if x <= 15 else 0)
    frame['Calcium'] = frame['Calcium'].apply(lambda x: 1 if x <= 75 else 0)
    frame['Magnesium'] = frame['Magnesium'].apply(lambda x: 1 if x <= 50 else 0)
    frame['Sodium'] = frame['Sodium'].apply(lambda x: 1 if x <= 200 else 0)
    frame['Potassium'] = frame['Potassium'].apply(lambda x: 1 if x <= 12 else 0)
    frame['Bicarbonate'] = frame['Bicarbonate'].apply(lambda x: 1 if x <= 350 else 0)
    frame['Carbonate'] = frame['Carbonate'].apply(lambda x: 1 if x <= 1 else 0)  # mean_value = 1
    le = LabelEncoder()
    frame["District"] = le.fit_transform(frame["District"])
    frame["Village"] = le.fit_transform(frame["Village"])
    le2 = LabelEncoder()
    frame["WQC"] = le2.fit_transform(frame["WQC"])

    # frame['hardness_temporal'] = frame['hardness_temporal'].apply(lambda x: 1)
    # frame['hardness_permanent'] = frame['hardness_permanent'].apply(lambda x: 1)
    # frame['co2_free'] = frame['co2_free'].apply(lambda x: 1)
    # frame['co2_depend'] = frame['co2_depend'].apply(lambda x: 1)
    # frame['co2_infiltrate'] = frame['co2_infiltrate'].apply(lambda x: 1)
    # frame['sio2'] = frame['sio2'].apply(lambda x: 1)
    # frame['is_drinkable'] = frame.apply(lambda x: 1 if ((x["Sodium"] == x["k"]) and (x["k"] == x["ca2"]) and
    #                                                     (x["ca2"] == x["mg2"]) and (x["mg2"] == x["fe3"]) and (
    #                                                             x["fe3"] == x["fe2"]) and (x["fe2"] == x["cl"])
    #                                                     and (x["cl"] == x["so4"]) and (x["so4"] == x["hco3"]) and (
    #                                                             x["hco3"] == x["co3"]) and (
    #                                                             x["co3"] == x["no2"]) and (
    #                                                             x["no2"] == x["no3"] and (x["no3"] == x["ph"])
    #                                                             and (x["ph"] == x["sio2"]) and (
    #                                                                     x["sio2"] == x["tds105"]) and (
    #                                                                     x["tds105"] == x["hardness_general"]) and (
    #                                                                     x["hardness_general"] == x["hardness_permanent"]
    #                                                             )
    #                                                             and (x["hardness_permanent"] == x["hardness_temporal"])
    #                                                             and (
    #                                                                     x["hardness_temporal"] == x["co2_free"]) and (
    #                                                                     x["co2_free"] == x["co2_depend"])
    #                                                             and (x["co2_depend"] == x["co2_infiltrate"]))
    #                                                     ) else 0, axis=1)
    return frame, le2


def get_indian_data(file_path):
    data = get_csv_data(file_path)
    describe(data)
    check_null(data)
    check_info(data)
    data = preprocess_indian_missing_data(data)
    data, encoder = transform_indian_features(data)
    return data, encoder


if __name__ == "__main__":
    path = "Indian dataSet/data/newDatafile.csv"
    data, label_encoder = get_indian_data(path)
    data.to_csv("Indian dataSet/data/processed_data.csv",  index="Index")
