import matplotlib.pyplot as plt
import seaborn as sns

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