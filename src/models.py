from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def train_random_forest(X_train, y_train, X_val, y_val):
    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_val)[:, 1]
    return clf, roc_auc_score(y_val, preds)


def train_adaboost(X_train, y_train, X_val, y_val):
    clf = AdaBoostClassifier(
        n_estimators=100, learning_rate=0.8, random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_val)[:, 1]
    return clf, roc_auc_score(y_val, preds)


def train_catboost(X_train, y_train, X_val, y_val):
    clf = CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=6,
        eval_metric='AUC',
        verbose=False,
        random_seed=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_val)[:, 1]
    return clf, roc_auc_score(y_val, preds)


def train_xgboost(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    preds = model.predict(dval)
    return model, roc_auc_score(y_val, preds)


def train_lgbm(X_train, y_train, X_val, y_val):
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'seed': 42
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    preds = model.predict(X_val)
    return model, roc_auc_score(y_val, preds)