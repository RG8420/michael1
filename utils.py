import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocessing_pipeline
fig_save_dir = "./results/plots/"


def plot_univariate_analysis(frame, cols):
    for col in cols:
        sns.scatterplot(data=frame, x=frame.index, y=frame[col])
        # plt.show()
        plt.savefig(fig_save_dir + "scatter_fig_" + col + ".png")


def plot_bivariate_analysis(frame, cols):
    for col in cols:
        a, b = col
        sns.scatterplot(data=frame, x=a, y=b)
        # plt.show()
        plt.savefig(fig_save_dir + "bivariate_fig_" + a + '_' + b + ".png")


def plot_outliers(frame, cols, com_col='wqc'):
    for col in cols:
        sns.boxplot(x=com_col, y=col, data=frame)
        # plt.show()
        plt.savefig(fig_save_dir + "outlier_fig_" + col + ".png")


if __name__ == "__main__":
    data_path = "./Vietnam dataset-phase-3/data-phase-3/daluong.xlsx"
    data = preprocessing_pipeline(data_path)
    print(data.head())
    unv_analysis_cols = ['na', 'cl', 'hco3', 'ph', 'wqi']
    plot_univariate_analysis(data, unv_analysis_cols)
    outlier_analysis_cols = ['ph', 'fe2', 'tds105', 'cl']
    plot_outliers(data, outlier_analysis_cols)
    biv_analysis_cols = [('k', 'cl'), ('k', 'mg2'), ('tds105', 'na'), ('hco3', 'co2_depend'), ('na', 'mg2')]
    plot_bivariate_analysis(data, biv_analysis_cols)
    print("Done!!")
