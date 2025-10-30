import pendulum
from airflow.decorators import dag, task
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
    
DATA_DIR = Path("/opt/airflow/data")  # общая папка в контейнере

@dag(
    schedule='@once',
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    tags=["SGM"],
    dag_id = 'v3'
)
def recsys_etl_pipeline():
    
    @task()
    def extract():
        path = DATA_DIR / "raw" / "cuprum_events.xlsx"
        df = pd.read_excel(path, sheet_name="Лист4")
        return df.to_json()
        
    @task()
    def transform(raw_json: str):
        df = pd.read_json(raw_json)
        df.rename(columns={"пол":"gender", "возраст":"age"}, inplace=True)
        df = df[df['tags'] != 'Секс']
        df[df['action_type'] == 'CLICKED']
        df.drop(columns=["esb_ehr_id","patientnet_ehr_id", "medialog_ehr_id","action_type"], inplace=True)
        df = filter_for_iteration_range(df)
        return df.to_json()
    
    @task()
    def load(raw_json: str):
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        out_path = DATA_DIR / "processed" / f"clicks_clean ({timestamp}).csv"
        df = pd.read_json(raw_json)
        df.to_csv(out_path, index=False)
    
    #def transform(data: pd.DataFrame):
    #    step1 = remove_duplicates(data)
    #    step2 = fill_missing_values(step1)
    #    return step2

    def filter_for_iteration_range(df, min=4, max=50):
        interaction_counts = df.groupby('ehr_id')['article_id'].count()
        users_in_range = set(interaction_counts[(interaction_counts >= min) & (interaction_counts <= max)].index)
        df_filtered = df[df['ehr_id'].isin(users_in_range)]
        return df_filtered

    # негерируем негативные сэмлы для работы модели
    def generate_negative_samples(df, all_articles, n_negatives=3):
        negatives = []

        for user_id, group in df.groupby('ehr_id'):
            clicked = set(group['article_id'])
            not_clicked = list(all_articles - clicked)
            sampled = np.random.choice(not_clicked, size=n_negatives, replace=False)

            for art in sampled:
                negatives.append({
                    "ehr_id": user_id,
                    "article_id": art,
                    "label": 0
                })
        return pd.DataFrame(negatives)
    
    @task()
    def build_top(raw_json: str, top_n: int = 10):
        """
        Строит топ статей по полу и возрастным группам и сохраняет в csv/pkl.
        """
        df = pd.read_json(raw_json)

        # Маппинг для читаемости
        gender_map = {1: "Женщины", 2: "Мужчины"}
        age_map = {0: "0–17", 1: "18–29", 2: "30–44", 3: "45–59", 4: "60+"}

        # Функция для группировки возраста
        def age_group(age):
            if age < 18: return 0
            elif age < 30: return 1
            elif age < 45: return 2
            elif age < 60: return 3
            else: return 4

        df["age_group"] = df["age"].apply(age_group)

        # Считаем количество кликов
        grouped = (
            df.groupby(["gender", "age_group", "article_id", "title", "url"])
            .size()
            .reset_index(name="clicks")
        )

        # Берём топ-N для каждой группы
        grouped["rank"] = grouped.groupby(["gender", "age_group"])["clicks"] \
            .rank(method="first", ascending=False)

        top_df = grouped[grouped["rank"] <= top_n].copy()

        # Приводим к читаемому виду
        top_df["gender"] = top_df["gender"].map(gender_map)
        top_df["age_group"] = top_df["age_group"].map(age_map)
        top_df = top_df[["gender", "age_group", "rank", "title", "url", "clicks"]]

        final = (
            top_df[['gender', 'age_group', 'rank', 'title', 'url']]
            .sort_values(['gender', 'age_group', 'rank'])
        )
        
        # Сохраняем
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        out_csv = DATA_DIR / "processed" / f"top_articles_readable ({timestamp}).csv"
        out_pkl = DATA_DIR / "processed" / f"top_articles_readable ({timestamp}).pkl"

        final.to_csv(out_csv, index=False)
        #final.to_pickle(out_pkl)

        return final.to_json()
    
    def remove_duplicates(data):
        feature_cols = data.columns.drop('customer_id').tolist()
        is_duplicated_features = data.duplicated(subset=feature_cols, keep=False)
        data = data[~is_duplicated_features].reset_index(drop=True)
        return data
    
    def fill_missing_values(data):
        cols_with_nans = data.isnull().sum()
        cols_with_nans = cols_with_nans[cols_with_nans > 0].index.drop('end_date')
        for col in cols_with_nans:
            if data[col].dtype in [float, int]:
                fill_value = data[col].mean()
            elif data[col].dtype == 'object':
                fill_value = data[col].mode().iloc[0]
            data[col] = data[col].fillna(fill_value)
        return data    

    raw = extract()
    clean = transform(raw)
    load(clean)
    build_top(clean, top_n = 20)
    
recsys_etl_pipeline()