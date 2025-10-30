import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

data = pd.read_excel("cuprum_3.xlsx", sheet_name="Лист4") #cuprum/Лист4/Лист2/Лист1
data

data.drop(columns=["esb_ehr_id","patientnet_ehr_id"], inplace=True)
data.rename(columns={"пол":"gender", "возраст":"age"}, inplace=True)
data = data[data['tags'] != 'Секс']
data

df = data[data['action_type'] == 'CLICKED']

df['weight'] = 1

# агрегируем, если нужно (например, несколько кликов одного юзера по одной статье)
df = df.groupby(['ehr_id', 'article_id'], as_index=False)['weight'].sum()

from scipy.sparse import coo_matrix

# создаём маппинги
user_ids = df['ehr_id'].unique().tolist()
item_ids = df['article_id'].unique().tolist()
user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {i: j for j, i in enumerate(item_ids)}

# строим COO-матрицу
rows = df['ehr_id'].map(user_map)
cols = df['article_id'].map(item_map)
data_weight = df['weight']

interactions = coo_matrix((data_weight, (rows, cols)),
                          shape=(len(user_ids), len(item_ids))).tocsr()


import implicit

# задаём гиперпараметры
factors = 50        # число латентных факторов
regularization = 0.01
iterations = 20
alpha = 40          # параметр масштабирования для implicit feedback

# модель
model = implicit.als.AlternatingLeastSquares(
    factors=factors,
    regularization=regularization,
    iterations=iterations,
)

# масштабируем веса (опционально)
# model.fit((interactions * alpha).astype('double'))
model.fit(interactions)


# Получаем два массива: индексы артиклов и их "скор"
user_idx = user_map[961420]
item_idxs, scores = model.recommend(user_idx, interactions[user_idx], N=10)

# Конвертируем индексы обратно в реальные article_id
rec_article_ids = [item_ids[i] for i in item_idxs]

# Если вам нужны и оценки — запакуем обратно в кортежи
recommendations = list(zip(rec_article_ids, scores))

print("Топ‑10 рекомендаций (article_id, score):")
for article_id, score in recommendations:
    print(f"{article_id} (score={score:.3f})")
    
    
# 1. Составляем словарь из всех article_id → title
id_to_title = dict(zip(data['article_id'], data['title']))

# 2. Для своего списка рекомендаций строим список названий
recommended_titles = [id_to_title[a_id] for a_id in rec_article_ids]

recommended_titles


# 1. Составляем маппинг для article_id -> title, url
id_to_title = dict(zip(data['article_id'], data['title']))
id_to_url = dict(zip(data['article_id'], data['url']))

# 2. Формируем таблицу рекомендаций
recs_df = pd.DataFrame({
    'article_id': rec_article_ids,
    'title': [id_to_title[a_id] for a_id in rec_article_ids],
    'url': [id_to_url[a_id] for a_id in rec_article_ids],
    'score': scores
})

# Сортируем по score, на всякий случай (ALS и так возвращает отсортированное)
recs_df = recs_df.sort_values('score', ascending=False).reset_index(drop=True)
recs_df.drop(columns=["score"], inplace=True)


recs_df


def recommend_for_user_als(user_ehr_id, N=10):
    """
    Вернуть рекомендации для одного пользователя:
    - user_ehr_id — ehr_id пользователя из данных
    - model — обученная ALS модель implicit
    - interactions — sparse матрица взаимодействий
    - user_map — словарь {ehr_id: user_idx}
    - item_ids — список всех article_id
    - data — исходный DataFrame с title и url
    - N — сколько рекомендаций вернуть

    Возвращает: DataFrame с article_id, title, url
    """

    # Проверим, что пользователь есть в маппинге
    #if user_ehr_id not in user_map:
    #    print(f"Пользователь {user_ehr_id} не найден.")
    #    return pd.DataFrame(columns=['article_id', 'title', 'url'])

    user_idx = user_map[user_ehr_id]

    # ALS рекомендации
    item_idxs, scores = model.recommend(user_idx, interactions[user_idx], N=N)

    # Индексы -> article_id
    rec_article_ids = [item_ids[i] for i in item_idxs]

    # Маппинги title + url
    id_to_title = dict(zip(data['article_id'], data['title']))
    id_to_url = dict(zip(data['article_id'], data['url']))

    # Финальный DataFrame
    recs_df = pd.DataFrame({
        'article_id': rec_article_ids,
        'title': [id_to_title.get(a_id, '') for a_id in rec_article_ids],
        'url': [id_to_url.get(a_id, '') for a_id in rec_article_ids],
        'score': scores
    }).sort_values('score', ascending=False).reset_index(drop=True)

    # Убираем score, если не нужен
    recs_df.drop(columns=['score'], inplace=True)

    return recs_df


