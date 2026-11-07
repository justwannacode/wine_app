import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

st.set_page_config(page_title='Подбор вин', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def test_curves(y, proba):
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    prec, rec, _ = metrics.precision_recall_curve(y, proba)
    return fpr, tpr, prec, rec


model = load_model()
config = load_config()
test = load_test()
X_test, y_test = test.drop(columns='target'), test['target']
FEATURES = list(X_test.columns)
proba_test = test_proba(model, X_test)

st.title('Модель отбора вин')

with st.sidebar:
    st.header('Настройки')
    threshold = st.slider('Порог решения', 0.0, 1.0, float(config['threshold']), 0.01, key='threshold')
    st.caption(f'Порог: {config["threshold"]:.3f}')

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка вина', 'Метрики', 'Предсказание', 'Что выучила модель'])

with tab_one:
    st.subheader('Введите результаты анализа')
    values = {}
    cols = st.columns(3)
    numeric = [f for f in FEATURES if f != 'is_red']
    for i, f in enumerate(numeric):
        lo, hi, med = float(X_test[f].min()), float(X_test[f].max()), float(X_test[f].median())
        values[f] = cols[i % 3].number_input(f, min_value=lo, max_value=hi, value=med, format='%.4f')
    wine_type = st.radio('Тип вина', ['белое', 'красное'], horizontal=True)
    values['is_red'] = int(wine_type == 'красное')

    wine = pd.DataFrame([values])[FEATURES]
    p = model.predict_proba(wine)[0, 1]
    c1, c2 = st.columns(2)
    c1.metric('Вероятность, что вино хорошее', f'{p:.1%}')
    c2.metric('Решение модели', 'В подборку' if p >= threshold else 'Не включать')

with tab_metrics:
    pred = (proba_test >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric('Precision', f'{precision:.3f}')
    c2.metric('Recall', f'{recall:.3f}')
    c3.metric('F1', f'{f1:.3f}')

    fpr, tpr, prec_curve, rec_curve = test_curves(y_test, proba_test)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['не хорошее', 'хорошее']).plot(
        ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Матрица ошибок')

    axes[1].plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, proba_test):.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].scatter(fp / (fp + tn), recall, s=120, color='red', zorder=3, label=f'порог {threshold:.2f}')
    axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')
    axes[1].legend()

    axes[2].plot(rec_curve, prec_curve, label=f'PR-AUC = {metrics.average_precision_score(y_test, proba_test):.3f}')
    axes[2].axhline(y_test.mean(), color='k', linestyle='--', label='случайная модель')
    if tp + fp > 0:
        axes[2].scatter(recall, precision, s=120, color='red', zorder=3, label=f'порог {threshold:.2f}')
    axes[2].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    axes[2].legend()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f'При текущем пороге в подборку попадает {tp + fp} вин из {len(y_test)}: '
    )

with tab_batch:
    st.subheader('Оценка списка вин из CSV')
    st.caption(f'Нужные столбцы: {", ".join(FEATURES)}. Порог берётся из сайдбара: {threshold:.2f}.')
    uploaded = st.file_uploader('Загрузите CSV-файл', type='csv')
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Не удалось прочитать файл: {e}')
            data = None
        if data is not None:
            missing = [f for f in FEATURES if f not in data.columns]
            if missing:
                st.error(f'В файле не хватает столбцов: {", ".join(missing)}')
            else:
                try:
                    result = data.copy()
                    result['proba_good'] = model.predict_proba(data[FEATURES])[:, 1]
                    result['in_selection'] = (result['proba_good'] >= threshold).astype(int)
                    st.dataframe(result)
                    st.write(f'В подборку попало {result["in_selection"].sum()} из {len(result)} вин.')
                    st.download_button('Скачать результат', result.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv')
                except Exception as e:
                    st.error(f'Не удалось посчитать предсказания: {e}')

with tab_coef:
    st.subheader('Коэффициенты логистической регрессии')
    coef = pd.Series(model[-1].coef_[0], index=model.feature_names_in_).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(coef.index, coef.values, color=np.where(coef.values > 0, 'tab:green', 'tab:red'))
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel('Коэффициент')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        f'Сильнее всего **за** хорошее вино: {", ".join(coef[coef > 0].sort_values(ascending=False).index[:3])}.\n\n'
        f'Сильнее всего **против**: {", ".join(coef[coef < 0].index[:3])}.\n\n')
