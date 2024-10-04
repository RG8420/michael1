import pandas as pd


def get_excel_data(data_path):
    frame = pd.read_excel(data_path)
    return frame


def get_csv_data(data_path):
    frame = pd.read_csv(data_path)
    return frame


if __name__ == "__main__":
    path = "./Vietnam dataset-phase-3/data-phase-3/daluong.xlsx"
    data = get_excel_data(path)
    print(data.columns)
    print("Done!!")
