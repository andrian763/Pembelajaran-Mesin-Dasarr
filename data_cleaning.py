import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_dataset(path: str): 
    df = pd.read_csv(path, na_values=["Unknown","N/A"])
    return df

def check_drop_duplicates(df: pd.DataFrame):
    is_duplicate = df.duplicated(subset=["id","age"])
    total_duplicates = is_duplicate.sum()
    df = df.drop_duplicates(subset=["id","age"])
    return df

def remove_missing_numeric_values(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    mask = df["age"] > 17
    df = df[mask]
    df = df.dropna(subset=numeric_columns)
    return df
    
def plot_histograms(f: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes = axes.ravel()
    cols = ["avg_glucose_level", "bmi"]
    colors = ['tab:blue', 'tab:orange']

    for col, color, ax in zip(cols, colors, axes):
        f[col].plot(kind='box', ax=ax, color=color, label=col, title=col)
        
    fig.tight_layout()
    plt.show()
    
def replace_outliers(df: pd.DataFrame):
    bmi_max = 50
    bmi_median = np.median(df["bmi"])
    print(bmi_median)
    avg_glucose_max = 180
    glucose_median = np.median(df["avg_glucose_level"])
    print(glucose_median)
    df["bmi"] = df["bmi"].apply(lambda x: bmi_median if x > bmi_max else x)
    df["avg_glucose_level"] = df["avg_glucose_level"].apply(lambda x: glucose_median if x > avg_glucose_max else x)
    return df

def remove_missing_categorical_values(df: pd.DataFrame):
    text_columns = df.select_dtypes(include=object).columns.tolist()
    smoking_status = df["smoking_status"]
    replacement_dict = {np.nan : "never smoked"}
    smoking_status = smoking_status.replace(replacement_dict)
    df["smoking_status"] = smoking_status
    return df
    
def clean_data(path: str):
    df = load_dataset(path)
    df = check_drop_duplicates(df)
    df = remove_missing_numeric_values(df)
    df = replace_outliers(df)
    df = remove_missing_categorical_values(df)
    return df

if __name__ == "__main__":
    data_path = "healthcare-dataset-stroke-data.csv" 
    cleaned_data = clean_data(path=data_path)
    plot_histograms(cleaned_data)
    cleaned_data.to_pickle("cleaned_data.pkl")
