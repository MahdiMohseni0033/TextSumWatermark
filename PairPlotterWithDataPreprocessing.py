import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


def ignore_future_warnings():
    """
    Ignore all future warnings.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)


def convert_categorical_columns_to_category(df):
    """
    Convert categorical columns to the category data type.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    categorical_columns = df.select_dtypes(include=['category', 'object']).columns
    df[categorical_columns] = df[categorical_columns].astype('category')


def plot_pairplot_and_save(df, save_path):
    """
    Plot pairplot of numerical columns using Seaborn and save it as a high-quality image.

    Args:
        df (pd.DataFrame): Input DataFrame containing numerical columns.
        save_path (str): Path to save the plot image.
    """
    sns.pairplot(df)
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    ignore_future_warnings()
    # Read the CSV file
    df = pd.read_csv('result.csv')

    # Handle missing values: Remove rows with NaN values
    df.dropna(inplace=True)

    WO_numerical_columns = ['WO_num_tokens_scored', 'WO_num_green_tokens', 'WO_green_fraction',
                            'WO_z_score', 'WO_p_value', 'WO_num_output_token']

    df_without_watermark = df[WO_numerical_columns].astype(float)
    save_path_without_watermark = 'all_parameter_without_watermark_pair_plot.png'
    plot_pairplot_and_save(df_without_watermark, save_path_without_watermark)

    W_numerical_columns = ['W_num_tokens_scored', 'W_num_green_tokens', 'W_green_fraction',
                           'W_z_score', 'W_p_value', 'W_num_output_token']

    df_with_watermark = df[W_numerical_columns].astype(float)
    save_path_with_watermark = 'all_parameter_with_watermark_pair_plot.png'
    plot_pairplot_and_save(df_with_watermark, save_path_with_watermark)
