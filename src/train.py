import argparse
from data_loader import DataLoader
from eda import plot_class_balance, plot_time_vs_amount, plot_roc_pr_curves
from models import (train_random_forest, train_adaboost,
                    train_catboost, train_xgboost, train_lgbm)
from sklearn.model_selection import train_test_split


def main(data_path, model_keys):
    df = DataLoader(data_path).load()
    DataLoader(data_path).summary(df)

    # EDA plots
    plot_class_balance(df)
    plot_time_vs_amount(df)

    # Split
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    runners = {
        'rf': train_random_forest,
        'ada': train_adaboost,
        'cat': train_catboost,
        'xgb': train_xgboost,
        'lgbm': train_lgbm
    }
    trained = {}

    for key in model_keys:
        fn = runners.get(key)
        if not fn:
            print(f"Unknown model: {key}")
            continue
        model, auc = fn(X_train, y_train, X_val, y_val)
        print(f"{key.upper()} Validation ROC-AUC: {auc:.4f}")
        trained[key] = model

    # Advanced curves
    plot_roc_pr_curves(trained, X_val, y_val)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Credit Card Fraud Detection')
    p.add_argument('--data-path', required=True)
    p.add_argument('--models', default='rf,ada,cat,xgb,lgbm')
    args = p.parse_args()
    main(args.data_path, args.models.split(','))