import pandas as pd
import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import re

def download_data():
    """
    Funkcja ściągająca dane z Kaggle do folderu.
    """
    api = KaggleApi()
    api.authenticate()

    dataset = "titanic"
    dataset_test = "wesleyhowe/titanic-labelled-test-set"
    download_folder = Path("data/titanic")
    zip_path = download_folder / "titanic.zip"
    download_folder.mkdir(parents=True, exist_ok=True)

    api.competition_download_files(dataset, path=str(download_folder))
    api.dataset_download_files(dataset_test, path=str(download_folder), unzip=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(str(download_folder))

    os.remove(zip_path)

def preprocess_data(file_path):
    """
    Funkcja do przetwarzania danych: usuwanie niepotrzebnych kolumn, wypełnianie brakujących wartości
    i inżynieria cech.
    """
    df = pd.read_csv(file_path)

    # Usuwanie kolumn Name, Ticket, Cabin
    df = df.drop(columns=["Name", "Ticket", "Cabin"])

    # Wypełnianie brakujących wartości
    df = df.fillna({"Embarked": "N", "Age": df["Age"].mean()})

    # Inżynieria cech
    df["Title"] = df["Name"].apply(lambda x: extract_title(x))
    pattern = r'([A-Za-z]+)(\d+)'
    matches = df['Cabin'].str.extractall(pattern)
    matches.reset_index(inplace=True)
    result = matches.pivot(index='level_0', columns='match', values=[0, 1])
    result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
    df = df.join(result[["0_0", "1_0"]])
    df["1_0"] = df["1_0"].astype(float)
    df = df.fillna({"0_0": "N", "1_0": df["1_0"].mean()})
    df["1_0"] = df["1_0"].astype(int)
    df = df.rename(columns={"0_0": "Deck", "1_0": "CabinNumber"})

    return df

def extract_title(name):
    """
    Funkcja wyciągająca tytuł z imienia i nazwiska pasażera.
    """
    match = re.search(r',\s*([\w\s]+)\.', name)
    return match.group(1) if match else None
    