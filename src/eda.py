import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def plot_class_balance(df, target='Class'):
    counts = df[target].value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()


def plot_time_vs_amount(df):
    fraud = df[df['Class']==1]
    plt.scatter(fraud['Time'], fraud['Amount'], alpha=0.4, c='red')
    plt.title('Fraud Amount over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amount')
    plt.show()


def plot_roc_pr_curves(models, X_val, y_val):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    plt.figure(figsize=(12,5))
    # ROC
    plt.subplot(1,2,1)
    for name, m in models.items():
        probs = m.predict_proba(X_val)[:,1]
        fpr, tpr, _ = roc_curve(y_val, probs)
        plt.plot(fpr, tpr, label=f"{name}(AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curves')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    # PR
    plt.subplot(1,2,2)
    for name, m in models.items():
        probs = m.predict_proba(X_val)[:,1]
        p, r, _ = precision_recall_curve(y_val, probs)
        plt.plot(r, p, label=name)
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()