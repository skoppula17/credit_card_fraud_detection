import argparse
from data_loader import DataLoader
from eda import (
    plot_class_balance,
    plot_time_vs_amount,
    plot_roc_pr_curves,
    plot_shap_summary
)
from models import (
    train_random_forest,
    train_adaboost,
    train_catboost,
    train_xgboost,
    train_lgbm
)
from sklearn.model_selection import train_test_split
import os

def main(data_path, models_list):
    # Load and summarize
    loader = DataLoader(data_path)
    df = loader.load()
    loader.summary(df)

    # EDA
    plot_class_balance(df)
    plot_time_vs_amount(df)

    # Prepare features
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Train each requested model
    runners = {
        'rf': train_random_forest,
        'ada': train_adaboost,
        'cat': train_catboost,
        'xgb': train_xgboost,
        'lgbm': train_lgbm
    }

    trained_models = {}
    for key in models_list:
        if key in runners:
            model, auc = runners[key](X_train, y_train, X_val, y_val)
            print(f"{key.upper()} Validation ROC-AUC: {auc:.4f}")
            trained_models[key] = model
        else:
            print(f"Unknown model: {key}")

    # Advanced plots
    plot_roc_pr_curves(trained_models, X_val, y_val)

    # SHAP summary for tree models
    for name, model in trained_models.items():
        if name in ('rf','ada','cat','xgb','lgbm'):
            print(f"Generating SHAP summary for {name} ...")
            plot_shap_summary(model, X_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Credit Card Fraud Detection'
    )
    parser.add_argument(
        '--data-path', required=True,
        help='Path to creditcard.csv'
    )
    parser.add_argument(
        '--models', default='rf,ada,cat,xgb,lgbm',
        help='Comma-separated list: rf,ada,cat,xgb,lgbm'
    )
    args = parser.parse_args()
    model_keys = args.models.split(',')
    main(args.data_path, model_keys)
