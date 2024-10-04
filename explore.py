from data import get_excel_data


def describe(frame):
    frame.describe()


def check_null(frame):
    print(frame.isna().sum())


def check_info(frame):
    frame.info()


if __name__ == "__main__":
    path = "./Vietnam dataset-phase-3/data-phase-3/daluong.xlsx"
    data = get_excel_data(path)
    print("Data Import Done!!")
    print("Describing Data...")
    describe(data)
    print("Describing Done!!")
    print("Checking Null Values..")
    check_null(data)
    print("Checking Null Done!!")
    print("Checking Data info...")
    check_info(data)
    print("Checking Info Done!!")
