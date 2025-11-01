# Content Article Recommendation System
 
Проект рекомендательной системы для медицинских статей с использованием различных подходов машинного обучения.
 
## О проекте
 
Система анализирует клики пользователей на статьи и строит персонализированные рекомендации на основе демографических данных (пол, возраст) и истории взаимодействий.
 
## Структура проекта
 
```
├── eda.ipynb                    # Exploratory Data Analysis
├── airflow/                     # ETL pipeline для обработки данных
│   └── dags/prepare_data.py    # DAG для извлечения, очистки и подготовки данных
├── top/                         # Простые рекомендации по популярности
│   ├── top_by_user_age_sex.ipynb
│   └── random_vs_top.py
├── rec_sys_tf_idf/             # Content-based рекомендации
│   └── rec_sys_tf_idf.ipynb
├── rec_sys_als/                # Collaborative filtering
│   └── rec_sys_als.ipynb
└── rec_sys_catboost/           # Ranking модель
    └── rec_sys_catboost.ipynb
```
 
## Airflow Pipeline
 
ETL процесс автоматизирован через Apache Airflow:
 
- **Extract**: Загрузка сырых данных из Excel
- **Transform**: Очистка данных, фильтрация пользователей по количеству взаимодействий (4-50)
- **Load**: Сохранение обработанных данных
- **Build Top**: Построение топовых рекомендаций по полу и возрастным группам
 
## Реализованные подходы
 
### 1. **Top по демографии**
Простые рекомендации на основе популярности статей среди похожих пользователей (пол + возрастная группа).
 
### 2. **TF-IDF (Content-Based)**
Рекомендации на основе содержания статей и тегов с использованием векторизации текста.
 
### 3. **ALS (Alternating Least Squares)**
Collaborative filtering подход с использованием библиотеки `implicit`. Матричная факторизация для поиска латентных связей между пользователями и статьями.
 
### 4. **CatBoost Ranker**
Learning-to-Rank модель с градиентным бустингом. Использует features пользователей и статей для ранжирования рекомендаций.


================================================================================

# Content Article Recommendation System
 
A recommendation system project for medical articles using various machine learning approaches.

## About the Project

The system analyzes user clicks on articles and builds personalized recommendations based on demographic data (gender, age) and interaction history.

## Project Structure
```
├── eda.ipynb                    # Exploratory Data Analysis
├── airflow/                     # ETL pipeline for data processing
│   └── dags/prepare_data.py    # DAG for data extraction, cleaning and preparation
├── top/                         # Simple popularity-based recommendations
│   ├── top_by_user_age_sex.ipynb
│   └── random_vs_top.py
├── rec_sys_tf_idf/             # Content-based recommendations
│   └── rec_sys_tf_idf.ipynb
├── rec_sys_als/                # Collaborative filtering
│   └── rec_sys_als.ipynb
└── rec_sys_catboost/           # Ranking model
    └── rec_sys_catboost.ipynb
```

## Airflow Pipeline

The ETL process is automated through Apache Airflow:
- **Extract**: Loading raw data from Excel
- **Transform**: Data cleaning, filtering users by interaction count (4-50)
- **Load**: Saving processed data
- **Build Top**: Building top recommendations by gender and age groups

## Implemented Approaches

### 1. Demographics-based Top
Simple recommendations based on article popularity among similar users (gender + age group).

### 2. TF-IDF (Content-Based)
Recommendations based on article content and tags using text vectorization.

### 3. ALS (Alternating Least Squares)
Collaborative filtering approach using the `implicit` library. Matrix factorization to find latent connections between users and articles.

### 4. CatBoost Ranker
Learning-to-Rank model with gradient boosting. Uses user and article features for ranking recommendations.