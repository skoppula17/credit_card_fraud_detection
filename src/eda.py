import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

sns.set_style('whitegrid')

def plot_class_balance(df, target='Class'):
    """
    Plots the distribution of the target classes.
    """
    counts = df[target].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title('Class Distribution (0 = legit, 1 = fraud)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def plot_time_vs_amount(df):
    """
    Scatter of fraudulent transaction amounts over time.
    """
    fraud = df[df['Class'] == 1]
    plt.figure(figsize=(8,4))
    plt.scatter(fraud['Time'], fraud['Amount'], alpha=0.4, c='red')
    plt.title('Fraudulent Transaction Amount over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amount')
    plt.show()

def plot_roc_pr_curves(models, X_val, y_val):
    """
    Given a dict of trained models, plot ROC and Precision-Recall curves.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    plt.figure(figsize=(14,6))

    # ROC
    plt.subplot(1,2,1)
    for name, model in models.items():
        probs = model.predict_proba(X_val)[:,1]
        fpr, tpr, _ = roc_curve(y_val, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # Precision-Recall
    plt.subplot(1,2,2)
    for name, model in models.items():
        probs = model.predict_proba(X_val)[:,1]
        p, r, _ = precision_recall_curve(y_val, probs)
        plt.plot(r, p, label=name)
    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

def plot_shap_summary(model, X_train):
    """
    Generate a SHAP summary plot for tree-based models.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
