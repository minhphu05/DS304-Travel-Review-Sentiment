# ========================== Core Utilities ==========================
import os
import re
import math
import random
import warnings
from datetime import datetime
from itertools import combinations
from collections import Counter
from typing import List, Tuple, Dict, Optional

# ========================== Scientific Computing ==========================
import numpy as np
import pandas as pd
import scipy
from scipy.stats import skew

# ========================== Visualization ==========================
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
import mplcursors
from matplotlib.colors import LinearSegmentedColormap

# ========================== Progress Tracking ==========================
from tqdm.notebook import tqdm
tqdm.pandas()

# ========================== Preprocessing ==========================
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OneHotEncoder,
    OrdinalEncoder, StandardScaler
)

# ========================== Model Selection & Evaluation ==========================
from sklearn.model_selection import (
    GridSearchCV, KFold, ParameterGrid, RandomizedSearchCV,
    StratifiedKFold, cross_val_predict, cross_val_score,
    train_test_split
)

from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, make_scorer, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from imblearn.metrics import classification_report_imbalanced

# ========================== Pipelines ==========================
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline, make_pipeline as imbalanced_make_pipeline

# ========================== Imbalanced Data Handling ==========================
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler

# ========================== Scikit-learn Models ==========================
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

# ========================== Gradient Boosting Models ==========================
import xgboost as xgb
from xgboost import XGBClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv

def train_dev_test (X, y):
    """
    Chia dữ liệu đầu vào thành 3 tập: train, dev (validation) và test, với tỉ lệ 60/20/20.

    Tham số:
        df (pd.DataFrame): DataFrame đầu vào, bắt buộc phải có cột 'label' là nhãn phân loại.

    Trả về:
        X_train (pd.DataFrame): Đặc trưng cho tập huấn luyện.
        y_train (pd.Series): Nhãn cho tập huấn luyện.
        X_dev (pd.DataFrame): Đặc trưng cho tập validation (dev).
        y_dev (pd.Series): Nhãn cho tập validation.
        X_test (pd.DataFrame): Đặc trưng cho tập kiểm tra (test).
        y_test (pd.Series): Nhãn cho tập kiểm tra.

    Ghi chú:
        - Tách theo tỉ lệ: Train (60%), Dev (20%), Test (20%).
        - Dữ liệu được **stratify** theo nhãn để đảm bảo phân phối lớp đồng đều giữa các tập.
        - In ra số lượng mỗi lớp trong từng tập để kiểm tra cân bằng.
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

    X_train, X_dev, y_train, y_dev = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42, shuffle=True)

    print("Train:", Counter(y_train))
    print("Dev:", Counter(y_dev))
    print("Test:", Counter(y_test))
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def evaluate_model(y_true, y_pred, y_proba=None, dataset_name='', target_names=None):
    """
    Đánh giá mô hình phân loại đa lớp hoặc nhị phân.

    Tham số:
        y_true: Nhãn thật
        y_pred: Nhãn dự đoán
        y_proba: Xác suất (nếu có) - không dùng cho đa lớp mặc định
        dataset_name: Tên tập dữ liệu (hiển thị)
        target_names: Danh sách tên lớp ['NEG', 'NEU', 'POS'] để hiển thị confusion matrix

    """

    print(f"\n🔍 Evaluation on {dataset_name} set:")

    # Các chỉ số macro (đa lớp)
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision (macro):", round(precision_score(y_true, y_pred, average='macro'), 4))
    print("Recall (macro)   :", round(recall_score(y_true, y_pred, average='macro'), 4))
    print("F1-score (macro) :", round(f1_score(y_true, y_pred, average='macro'), 4))

    # ROC AUC chỉ phù hợp với nhị phân hoặc xử lý đặc biệt → nên tạm bỏ trong đa lớp

    # Báo cáo chi tiết
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    
def optimize_threshold_by_f1(model, X_test, y_test, start=0.05, end=0.95, step=0.01, plot=True):
    """
    Tối ưu ngưỡng phân loại để đạt F1-score cao nhất.

    Parameters:
        model: Mô hình đã huấn luyện, phải có phương thức `predict_proba`.
        X_test (DataFrame or array): Dữ liệu kiểm tra.
        y_test (Series or array): Nhãn thật.
        start (float): Ngưỡng bắt đầu thử.
        end (float): Ngưỡng kết thúc thử.
        step (float): Bước nhảy của ngưỡng.
        plot (bool): Có hiển thị biểu đồ không.

    Returns:
        best_thresh (float): Ngưỡng phân loại tốt nhất.
        best_f1 (float): F1-score cao nhất tương ứng.
    """
    # Lấy xác suất của lớp 1
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(start, end, step)
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"✅ Best threshold: {best_thresh:.2f} with F1-score: {best_f1:.4f}")

    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(thresholds, f1_scores, marker='o')
        plt.xlabel("Threshold")
        plt.ylabel("F1-score")
        plt.title("Tối ưu threshold phân loại")
        plt.grid(True)
        plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best: {best_thresh:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return best_thresh, best_f1


def tune_logistic_regression_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict, pd.DataFrame]:
    """
    Tune Logistic Regression hyperparameters sử dụng GridSearchCV.
    """

    # Grid nhỏ gọn
    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear"],  # đảm bảo hỗ trợ cả l1 và l2
    }

    model = LogisticRegression(max_iter=1000, random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df

def tune_lightgbm_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict, pd.DataFrame]:
    """
    Tune LightGBM hyperparameters sử dụng GridSearchCV.
    """

    # Grid đơn giản, gọn nhẹ
    param_grid = {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [-1, 5, 10]
    }

    model = LGBMClassifier(verbose=-1, random_state=random_state, n_jobs=-1)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df

def tune_gaussian_nb_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5
) -> Tuple[Dict, pd.DataFrame]:
    """
    Tune GaussianNB hyperparameter (var_smoothing) sử dụng GridSearchCV.
    
    Returns:
        best_params: dict of best parameters
        results_df: DataFrame of GridSearchCV results
    """
    param_grid = {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    model = GaussianNB()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df

def tune_xgboost_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict, pd.DataFrame]:
    """
    Tune XGBoost hyperparameters sử dụng GridSearchCV.
    
    Trả về:
        best_params: dict of best parameters
        results_df: DataFrame of GridSearchCV results
    """

    # Grid nhỏ gọn, hiệu quả
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6, 10],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    model = XGBClassifier(
        objective='binary:logistic' if y.nunique() == 2 else 'multi:softprob',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df

def tune_random_forest_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict, pd.DataFrame]:
    """
    Tune Random Forest hyperparameters sử dụng GridSearchCV/
    
    Trả về:
        best_params: dict of best parameters
        results_df: DataFrame of GridSearchCV results
    """
    param_grid = {
        'n_estimators': [100],                 # Giữ 1 giá trị phổ biến
        'max_depth': [None, 10],               # Không giới hạn hoặc vừa phải
        'min_samples_split': [2],              # Mặc định
        'min_samples_leaf': [1],               # Mặc định
        'max_features': ['sqrt']               # Hiệu quả với cây
    }

    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df

def tune_svm_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict, pd.DataFrame]:
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    ]

    model = SVC(random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df


def tune_knn_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict, pd.DataFrame]:
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    model = KNeighborsClassifier()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df


def tune_catboost_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict, pd.DataFrame]:
    param_grid = {
        'iterations': [100, 300],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 6, 8]
    }

    model = CatBoostClassifier(
        verbose=0,                 # Không in log trong quá trình huấn luyện
        random_state=random_state,
        task_type='CPU'           # Đảm bảo tương thích không cần GPU
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df


def tune_decision_tree_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict, pd.DataFrame]:
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = DecisionTreeClassifier(random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df

def tune_mlp_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict, pd.DataFrame]:
    param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001],  # L2 regularization
        'learning_rate': ['constant', 'adaptive']
    }

    model = MLPClassifier(max_iter=300, random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, average="binary" if y.nunique() == 2 else "macro")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    results_df = pd.DataFrame(grid.cv_results_)

    return best_params, results_df