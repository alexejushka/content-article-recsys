import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
import pickle

def age_group(age):
    """Преобразование возраста в группу"""
    if age < 18:
        return 0  # '0-17'
    elif age < 30:
        return 1  # '18-29'
    elif age < 45:
        return 2  # '30-44'
    elif age < 60:
        return 3  # '45-59'
    else:
        return 4  # '60+'


def prepare_recommendation_systems(clicked_df: pd.DataFrame, n_top: int = 10) -> Dict:
    """
    Подготовка всех трех систем рекомендаций:
    1. Сегментные топы (по полу и возрасту)
    2. Общий топ для всех
    3. Рандом (для сравнения)
    
    Returns:
        Dict с топами для каждой системы
    """
    
    # 1. СЕГМЕНТНЫЕ ТОПЫ (по полу и возрасту)
    grouped = (
        clicked_df.groupby(['gender', 'age_group', 'article_id'])
        .size()
        .reset_index(name='clicks')
    )
    
    top_articles_by_segment = (
        grouped
        .sort_values(['gender', 'age_group', 'clicks'], ascending=[True, True, False])
        .groupby(['gender', 'age_group'])['article_id']
        .apply(lambda x: x.head(n_top).tolist())
        .to_dict()
    )
    
    # 2. ОБЩИЙ ТОП (без сегментации)
    global_top = (
        clicked_df.groupby('article_id')
        .size()
        .reset_index(name='clicks')
        .sort_values('clicks', ascending=False)
        .head(n_top)['article_id']
        .tolist()
    )
    
    # 3. Информация о статьях
    articles_info = clicked_df[['article_id', 'title', 'url']].drop_duplicates()
    
    return {
        'segment_tops': top_articles_by_segment,
        'global_top': global_top,
        'articles_info': articles_info,
        'all_articles': list(clicked_df['article_id'].unique())
    }


def evaluate_all_methods(clicked_df: pd.DataFrame, 
                        recommendation_systems: Dict,
                        n_recommendations: int = 10,
                        n_random_iterations: int = 100) -> Dict:
    """
    Сравнение всех трех методов:
    1. Сегментные топы
    2. Общий топ
    3. Рандом
    """
    
    segment_tops = recommendation_systems['segment_tops']
    global_top = recommendation_systems['global_top'][:n_recommendations]
    all_articles = recommendation_systems['all_articles']
    
    # Инициализация метрик
    metrics = {
        'segment': {'hits': 0, 'total_clicks': [], 'precisions': []},
        'global': {'hits': 0, 'total_clicks': [], 'precisions': []},
        'random': {'hits': [], 'total_clicks': [], 'precisions': []}
    }
    
    # Уникальные пользователи
    users = clicked_df[['ehr_id', 'gender', 'age_group']].drop_duplicates()
    total_users = len(users)
    
    print(f"Оцениваем {total_users} пользователей по трем методам...")
    
    for _, user_row in users.iterrows():
        user_id = user_row['ehr_id']
        gender = user_row['gender']
        age_grp = user_row['age_group']
        
        # Статьи, которые кликнул пользователь
        user_clicks = set(clicked_df[clicked_df['ehr_id'] == user_id]['article_id'].unique())
        n_user_clicks = len(user_clicks)
        
        # 1. СЕГМЕНТНЫЕ РЕКОМЕНДАЦИИ
        segment_recs = segment_tops.get((gender, age_grp), [])[:n_recommendations]
        segment_recs_set = set(segment_recs)
        segment_matches = user_clicks & segment_recs_set
        
        if segment_matches:
            metrics['segment']['hits'] += 1
        metrics['segment']['total_clicks'].append(len(segment_matches))
        if len(segment_recs) > 0:
            metrics['segment']['precisions'].append(len(segment_matches) / len(segment_recs))
        
        # 2. ОБЩИЙ ТОП
        global_recs_set = set(global_top)
        global_matches = user_clicks & global_recs_set
        
        if global_matches:
            metrics['global']['hits'] += 1
        metrics['global']['total_clicks'].append(len(global_matches))
        if len(global_top) > 0:
            metrics['global']['precisions'].append(len(global_matches) / len(global_top))
        
        # 3. РАНДОМНЫЕ РЕКОМЕНДАЦИИ (усредняем по итерациям)
        user_random_hits = []
        user_random_clicks = []
        user_random_precisions = []
        
        for _ in range(n_random_iterations):
            random_recs = random.sample(all_articles, min(n_recommendations, len(all_articles)))
            random_recs_set = set(random_recs)
            random_matches = user_clicks & random_recs_set
            
            user_random_hits.append(1 if random_matches else 0)
            user_random_clicks.append(len(random_matches))
            user_random_precisions.append(len(random_matches) / len(random_recs))
        
        metrics['random']['hits'].append(np.mean(user_random_hits))
        metrics['random']['total_clicks'].append(np.mean(user_random_clicks))
        metrics['random']['precisions'].append(np.mean(user_random_precisions))
    
    # Расчет финальных метрик
    results = {}
    
    # Сегментные топы
    results['segment'] = {
        'hit_rate': metrics['segment']['hits'] / total_users,
        'avg_clicks_in_top': np.mean(metrics['segment']['total_clicks']),
        'median_clicks_in_top': np.median(metrics['segment']['total_clicks']),
        'precision': np.mean(metrics['segment']['precisions']),
        'users_with_hits': metrics['segment']['hits']
    }
    
    # Общий топ
    results['global'] = {
        'hit_rate': metrics['global']['hits'] / total_users,
        'avg_clicks_in_top': np.mean(metrics['global']['total_clicks']),
        'median_clicks_in_top': np.median(metrics['global']['total_clicks']),
        'precision': np.mean(metrics['global']['precisions']),
        'users_with_hits': metrics['global']['hits']
    }
    
    # Рандом
    results['random'] = {
        'hit_rate': np.mean(metrics['random']['hits']),
        'avg_clicks_in_top': np.mean(metrics['random']['total_clicks']),
        'median_clicks_in_top': np.median(metrics['random']['total_clicks']),
        'precision': np.mean(metrics['random']['precisions']),
        'users_with_hits': int(np.sum(metrics['random']['hits']))
    }
    
    # Улучшения относительно рандома
    for method in ['segment', 'global']:
        results[f'{method}_vs_random'] = {
            'hit_rate_improvement': (results[method]['hit_rate'] / results['random']['hit_rate'] - 1) * 100,
            'avg_clicks_improvement': (results[method]['avg_clicks_in_top'] / results['random']['avg_clicks_in_top'] - 1) * 100,
            'precision_improvement': (results[method]['precision'] / results['random']['precision'] - 1) * 100
        }
    
    # Сравнение сегментного и глобального
    results['segment_vs_global'] = {
        'hit_rate_improvement': (results['segment']['hit_rate'] / results['global']['hit_rate'] - 1) * 100,
        'avg_clicks_improvement': (results['segment']['avg_clicks_in_top'] / results['global']['avg_clicks_in_top'] - 1) * 100,
        'precision_improvement': (results['segment']['precision'] / results['global']['precision'] - 1) * 100
    }
    
    return results


def calculate_precision_at_k_all_methods(clicked_df: pd.DataFrame,
                                         recommendation_systems: Dict,
                                         k_values: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    Расчет Precision@K для всех трех методов
    """
    segment_tops = recommendation_systems['segment_tops']
    global_top = recommendation_systems['global_top']
    all_articles = recommendation_systems['all_articles']
    
    results = []
    users = clicked_df[['ehr_id', 'gender', 'age_group']].drop_duplicates()
    
    for k in k_values:
        segment_precisions = []
        global_precisions = []
        random_precisions = []
        
        for _, user_row in users.iterrows():
            user_id = user_row['ehr_id']
            gender = user_row['gender']
            age_grp = user_row['age_group']
            
            user_clicks = set(clicked_df[clicked_df['ehr_id'] == user_id]['article_id'].unique())
            
            # Сегментные рекомендации
            segment_recs = segment_tops.get((gender, age_grp), [])[:k]
            if segment_recs:
                segment_precision = len(set(segment_recs) & user_clicks) / k
                segment_precisions.append(segment_precision)
            
            # Глобальный топ
            global_recs = global_top[:k]
            if global_recs:
                global_precision = len(set(global_recs) & user_clicks) / k
                global_precisions.append(global_precision)
            
            # Рандомные рекомендации (усредняем)
            random_precision_iter = []
            for _ in range(100):
                random_recs = random.sample(all_articles, min(k, len(all_articles)))
                random_precision = len(set(random_recs) & user_clicks) / k
                random_precision_iter.append(random_precision)
            random_precisions.append(np.mean(random_precision_iter))
        
        results.append({
            'K': k,
            'Precision@K_segment': np.mean(segment_precisions),
            'Precision@K_global': np.mean(global_precisions),
            'Precision@K_random': np.mean(random_precisions),
            'Segment_vs_Random_%': ((np.mean(segment_precisions) / np.mean(random_precisions)) - 1) * 100,
            'Global_vs_Random_%': ((np.mean(global_precisions) / np.mean(random_precisions)) - 1) * 100,
            'Segment_vs_Global_%': ((np.mean(segment_precisions) / np.mean(global_precisions)) - 1) * 100
        })
    
    return pd.DataFrame(results)


def analyze_coverage_all_methods(clicked_df: pd.DataFrame,
                                 recommendation_systems: Dict,
                                 n_recommendations: int = 10) -> Dict:
    """
    Сравнение покрытия каталога для всех методов
    """
    segment_tops = recommendation_systems['segment_tops']
    global_top = recommendation_systems['global_top'][:n_recommendations]
    all_articles = set(recommendation_systems['all_articles'])
    n_articles = len(all_articles)
    
    # 1. Покрытие сегментными топами
    segment_articles = set()
    for articles_list in segment_tops.values():
        segment_articles.update(articles_list[:n_recommendations])
    segment_coverage = len(segment_articles) / n_articles
    
    # 2. Покрытие глобальным топом
    global_coverage = len(set(global_top)) / n_articles
    
    # 3. Покрытие рандомом (симуляция)
    n_segments = len(segment_tops)  # Количество сегментов
    simulated_coverage = []
    
    for _ in range(100):
        random_articles = set()
        # Симулируем рандомные рекомендации для каждого сегмента
        for _ in range(n_segments):
            random_sample = random.sample(list(all_articles), 
                                        min(n_recommendations, n_articles))
            random_articles.update(random_sample)
        simulated_coverage.append(len(random_articles) / n_articles)
    
    # Дополнительные метрики разнообразия
    # Gini коэффициент для оценки неравномерности распределения
    def calculate_gini(article_counts):
        sorted_counts = sorted(article_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        return (2 * np.sum((np.arange(1, n+1) * sorted_counts))) / (n * np.sum(sorted_counts)) - (n + 1) / n
    
    # Подсчет частот для сегментных топов
    segment_freq = {}
    for articles_list in segment_tops.values():
        for article in articles_list[:n_recommendations]:
            segment_freq[article] = segment_freq.get(article, 0) + 1
    
    segment_gini = calculate_gini(list(segment_freq.values())) if segment_freq else 0
    
    return {
        'segment_coverage': segment_coverage,
        'global_coverage': global_coverage,
        'random_coverage': np.mean(simulated_coverage),
        'segment_unique_articles': len(segment_articles),
        'global_unique_articles': len(set(global_top)),
        'segment_gini': segment_gini,
        'concentration_ratio_segment_vs_random': segment_coverage / np.mean(simulated_coverage),
        'concentration_ratio_global_vs_random': global_coverage / np.mean(simulated_coverage)
    }


def statistical_significance_test(clicked_df: pd.DataFrame,
                                 recommendation_systems: Dict,
                                 n_recommendations: int = 10,
                                 n_bootstrap: int = 1000) -> Dict:
    """
    Bootstrap тест для статистической значимости различий между методами
    """
    segment_tops = recommendation_systems['segment_tops']
    global_top = recommendation_systems['global_top'][:n_recommendations]
    all_articles = recommendation_systems['all_articles']
    
    users = clicked_df[['ehr_id', 'gender', 'age_group']].drop_duplicates()
    
    segment_hit_rates = []
    global_hit_rates = []
    random_hit_rates = []
    
    for _ in range(n_bootstrap):
        # Сэмплируем пользователей с возвратом
        sampled_users = users.sample(n=len(users), replace=True)
        
        segment_hits = 0
        global_hits = 0
        random_hits = 0
        
        for _, user_row in sampled_users.iterrows():
            user_clicks = set(clicked_df[clicked_df['ehr_id'] == user_row['ehr_id']]['article_id'])
            
            # Сегментные
            segment_recs = set(segment_tops.get(
                (user_row['gender'], user_row['age_group']), [])[:n_recommendations])
            if user_clicks & segment_recs:
                segment_hits += 1
            
            # Глобальные
            global_recs = set(global_top)
            if user_clicks & global_recs:
                global_hits += 1
            
            # Рандомные
            random_recs = set(random.sample(all_articles, min(n_recommendations, len(all_articles))))
            if user_clicks & random_recs:
                random_hits += 1
        
        segment_hit_rates.append(segment_hits / len(sampled_users))
        global_hit_rates.append(global_hits / len(sampled_users))
        random_hit_rates.append(random_hits / len(sampled_users))
    
    # Считаем различия и p-values
    segment_vs_random = np.array(segment_hit_rates) - np.array(random_hit_rates)
    global_vs_random = np.array(global_hit_rates) - np.array(random_hit_rates)
    segment_vs_global = np.array(segment_hit_rates) - np.array(global_hit_rates)
    
    return {
        'segment_vs_random': {
            'mean_diff': np.mean(segment_vs_random),
            'ci_95': [np.percentile(segment_vs_random, 2.5), np.percentile(segment_vs_random, 97.5)],
            'p_value': np.mean(segment_vs_random <= 0),
            'significant': np.mean(segment_vs_random <= 0) < 0.05
        },
        'global_vs_random': {
            'mean_diff': np.mean(global_vs_random),
            'ci_95': [np.percentile(global_vs_random, 2.5), np.percentile(global_vs_random, 97.5)],
            'p_value': np.mean(global_vs_random <= 0),
            'significant': np.mean(global_vs_random <= 0) < 0.05
        },
        'segment_vs_global': {
            'mean_diff': np.mean(segment_vs_global),
            'ci_95': [np.percentile(segment_vs_global, 2.5), np.percentile(segment_vs_global, 97.5)],
            'p_value': np.mean(segment_vs_global <= 0),
            'significant': np.mean(segment_vs_global <= 0) < 0.05
        }
    }


def print_full_comparison_report(clicked_df: pd.DataFrame,
                                n_recommendations: int = 10):
    """
    Полный отчет сравнения всех трех методов
    """
    # Подготовка систем рекомендаций
    print("Подготовка систем рекомендаций...")
    rec_systems = prepare_recommendation_systems(clicked_df, n_recommendations)
    
    print("=" * 80)
    print("СРАВНЕНИЕ ТРЕХ МЕТОДОВ РЕКОМЕНДАЦИЙ")
    print("=" * 80)
    print(f"1. Сегментные топы (пол × возраст)")
    print(f"2. Общий топ-{n_recommendations} для всех")
    print(f"3. Случайные рекомендации")
    print("=" * 80)
    
    # 1. ОСНОВНЫЕ МЕТРИКИ
    print("\n📊 ОСНОВНЫЕ МЕТРИКИ")
    print("-" * 60)
    metrics = evaluate_all_methods(clicked_df, rec_systems, n_recommendations)
    
    # Таблица с метриками
    methods = ['segment', 'global', 'random']
    method_names = ['Сегментные топы', 'Общий топ', 'Рандом']
    
    print(f"\n{'Метод':<20} {'Hit Rate':<12} {'Avg Clicks':<12} {'Precision':<12} {'Users w/hits':<12}")
    print("-" * 68)
    
    for method, name in zip(methods, method_names):
        print(f"{name:<20} "
              f"{metrics[method]['hit_rate']:<12.4f} "
              f"{metrics[method]['avg_clicks_in_top']:<12.4f} "
              f"{metrics[method]['precision']:<12.4f} "
              f"{metrics[method]['users_with_hits']:<12.0f}")
    
    # Улучшения
    print("\n📈 УЛУЧШЕНИЯ ОТНОСИТЕЛЬНО БАЗОВЫХ МЕТОДОВ")
    print("-" * 60)
    print(f"\nСегментные топы vs Рандом:")
    print(f"  Hit Rate: +{metrics['segment_vs_random']['hit_rate_improvement']:.1f}%")
    print(f"  Avg clicks: +{metrics['segment_vs_random']['avg_clicks_improvement']:.1f}%")
    print(f"  Precision: +{metrics['segment_vs_random']['precision_improvement']:.1f}%")
    
    print(f"\nОбщий топ vs Рандом:")
    print(f"  Hit Rate: +{metrics['global_vs_random']['hit_rate_improvement']:.1f}%")
    print(f"  Avg clicks: +{metrics['global_vs_random']['avg_clicks_improvement']:.1f}%")
    print(f"  Precision: +{metrics['global_vs_random']['precision_improvement']:.1f}%")
    
    print(f"\nСегментные топы vs Общий топ:")
    hit_rate_diff = metrics['segment_vs_global']['hit_rate_improvement']
    clicks_diff = metrics['segment_vs_global']['avg_clicks_improvement']
    precision_diff = metrics['segment_vs_global']['precision_improvement']
    
    print(f"  Hit Rate: {hit_rate_diff:+.1f}%")
    print(f"  Avg clicks: {clicks_diff:+.1f}%")
    print(f"  Precision: {precision_diff:+.1f}%")
    
    # 2. PRECISION@K
    print("\n📏 PRECISION@K ДЛЯ РАЗНЫХ K")
    print("-" * 60)
    precision_df = calculate_precision_at_k_all_methods(clicked_df, rec_systems)
    
    # Форматированный вывод
    print(f"\n{'K':<5} {'Segment':<12} {'Global':<12} {'Random':<12} {'Seg vs Rand':<15} {'Seg vs Glob':<15}")
    print("-" * 71)
    for _, row in precision_df.iterrows():
        print(f"{row['K']:<5} "
              f"{row['Precision@K_segment']:<12.6f} "
              f"{row['Precision@K_global']:<12.6f} "
              f"{row['Precision@K_random']:<12.6f} "
              f"{row['Segment_vs_Random_%']:+<15.1f}% "
              f"{row['Segment_vs_Global_%']:+<15.1f}%")
    
    # 3. ПОКРЫТИЕ КАТАЛОГА
    print("\n📚 ПОКРЫТИЕ КАТАЛОГА СТАТЕЙ")
    print("-" * 60)
    coverage = analyze_coverage_all_methods(clicked_df, rec_systems, n_recommendations)
    
    print(f"Сегментные топы:")
    print(f"  Покрытие: {coverage['segment_coverage']:.4f} ({coverage['segment_unique_articles']} статей)")
    print(f"  Gini коэффициент: {coverage['segment_gini']:.4f}")
    
    print(f"\nОбщий топ:")
    print(f"  Покрытие: {coverage['global_coverage']:.4f} ({coverage['global_unique_articles']} статей)")
    
    print(f"\nРандом (симуляция):")
    print(f"  Покрытие: {coverage['random_coverage']:.4f}")
    
    print(f"\nКоэффициенты концентрации (меньше = более сконцентрировано):")
    print(f"  Сегменты/Рандом: {coverage['concentration_ratio_segment_vs_random']:.2f}x")
    print(f"  Общий топ/Рандом: {coverage['concentration_ratio_global_vs_random']:.2f}x")
    
    # 4. СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ
    print("\n🔬 СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ (Bootstrap, n=1000)")
    print("-" * 60)
    
    significance = statistical_significance_test(clicked_df, rec_systems, n_recommendations)
    
    for comparison, label in [
        ('segment_vs_random', 'Сегментные топы vs Рандом'),
        ('global_vs_random', 'Общий топ vs Рандом'),
        ('segment_vs_global', 'Сегментные топы vs Общий топ')
    ]:
        result = significance[comparison]
        print(f"\n{label}:")
        print(f"  Средняя разница: {result['mean_diff']:.4f}")
        print(f"  95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
        print(f"  p-value: {result['p_value']:.4f}")
        
        if result['significant']:
            print(f"  ✅ Статистически значимо (p < 0.05)")
        else:
            print(f"  ❌ Статистически НЕ значимо")
    
    # 5. ВЫВОДЫ
    print("\n" + "=" * 80)
    print("💡 ВЫВОДЫ")
    print("=" * 80)
    
    # Определяем лучший метод
    best_method = max(methods[:2], key=lambda m: metrics[m]['hit_rate'])
    best_name = method_names[methods.index(best_method)]
    
    print(f"\n1. Лучший метод по Hit Rate: {best_name}")
    print(f"2. Улучшение относительно рандома: "
          f"{metrics[f'{best_method}_vs_random']['hit_rate_improvement']:.1f}%")
    
    if abs(metrics['segment_vs_global']['hit_rate_improvement']) < 5:
        print("3. Сегментация и общий топ показывают сопоставимые результаты (разница < 5%)")
        print("   → Можно начать с общего топа (проще реализация)")
    elif metrics['segment_vs_global']['hit_rate_improvement'] > 0:
        print("3. Сегментация дает заметное улучшение над общим топом")
        print("   → Рекомендуется использовать сегментные топы")
    else:
        print("3. Общий топ работает лучше сегментации")
        print("   → Возможно, недостаточно данных для качественной сегментации")
    
    print("\n" + "=" * 80)
    
    return rec_systems, metrics


def save_recommendations(rec_systems: Dict, output_prefix: str = 'recommendations'):
    """
    Сохранение всех систем рекомендаций в файлы
    """
    # Сохраняем сегментные топы
    with open(f'{output_prefix}_segment.pkl', 'wb') as f:
        pickle.dump(rec_systems['segment_tops'], f)
    
    # Сохраняем общий топ
    with open(f'{output_prefix}_global.pkl', 'wb') as f:
        pickle.dump(rec_systems['global_top'], f)
    
    # Сохраняем информацию о статьях
    rec_systems['articles_info'].to_csv(f'{output_prefix}_articles_info.csv', index=False)
    
    print(f"✅ Рекомендации сохранены с префиксом '{output_prefix}'")


# Функции для использования рекомендаций
def recommend_by_segment(gender: int, age: int, rec_systems: Dict, n: int = 10) -> pd.DataFrame:
    """Рекомендации на основе сегментации"""
    group = age_group(age)
    segment_tops = rec_systems['segment_tops']
    articles_info = rec_systems['articles_info']
    
    ids = segment_tops.get((gender, group), [])[:n]
    recs = articles_info[articles_info['article_id'].isin(ids)]
    recs['article_id'] = pd.Categorical(recs['article_id'], categories=ids, ordered=True)
    recs = recs.sort_values('article_id')
    return recs[['article_id', 'title', 'url']].reset_index(drop=True)


def recommend_global(rec_systems: Dict, n: int = 10) -> pd.DataFrame:
    """Рекомендации на основе общего топа"""
    global_top = rec_systems['global_top'][:n]
    articles_info = rec_systems['articles_info']
    
    recs = articles_info[articles_info['article_id'].isin(global_top)]
    recs['article_id'] = pd.Categorical(recs['article_id'], categories=global_top, ordered=True)
    recs = recs.sort_values('article_id')
    return recs[['article_id', 'title', 'url']].reset_index(drop=True)


def recommend_random(rec_systems: Dict, n: int = 10) -> pd.DataFrame:
    """Случайные рекомендации"""
    all_articles = rec_systems['all_articles']
    articles_info = rec_systems['articles_info']
    
    random_ids = random.sample(all_articles, min(n, len(all_articles)))
    recs = articles_info[articles_info['article_id'].isin(random_ids)]
    return recs[['article_id', 'title', 'url']].reset_index(drop=True)


# Пример использования
if __name__ == "__main__":
    # Загружаем данные
    data = pd.read_excel("../cuprum_3.xlsx", sheet_name="Лист4")
    data.drop(columns=["esb_ehr_id","patientnet_ehr_id"], inplace=True)
    data.rename(columns={"пол":"gender", "возраст":"age"}, inplace=True)
    #data = data[data['tags'] != 'Секс']
    data['age_group'] = data['age'].apply(age_group)
    clicked = data[data['action_type'] == 'CLICKED']
    
    # Строим систему
    print_full_comparison_report(clicked)
    #rec_systems = prepare_recommendation_systems(clicked)
    #evaluate_all_methods(rec_systems)
    #print_full_comparison_report(rec_systems)
    
    
    # Пример: показать рекомендации
    #print("Сегментные:", recommend_segment(rec_systems, gender="М", age=35, n=5))
    #print("Глобальные:", recommend_global(rec_systems, n=5))
    #print("Рандомные:", recommend_random(rec_systems, n=5))
    # Запускаем