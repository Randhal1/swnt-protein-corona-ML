#!python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_params():
    accuracy = np.loadtxt("accuracy.csv")
    auc = np.loadtxt("auc.csv")
    fscore = np.loadtxt("f1-score.csv")
    precision = np.loadtxt("precision.csv")
    recall = np.loadtxt("recall.csv")

    fig, ax = plt.subplots(figsize=[12,8])

    def plot_lines(var, var_name, color):
        ax.plot(var, label=var_name+f" {np.mean(var):.2f}$\pm${np.std(var):.2f}", color=color, lw=2)
        ax.plot(var, "p", color=color, lw=2)

    plot_lines(accuracy, "accuracy", "maroon")
    plot_lines(auc, "AUC", "navy")
    plot_lines(fscore, "F1 score", "forestgreen")
    plot_lines(precision, "precision", "orangered")
    plot_lines(recall, "recall", "dodgerblue")
    
    plt.xlabel("Iteration", size=14)
    plt.ylabel("Score", size=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.legend(loc="upper right")
    plt.xlim([0,100])
    plt.ylim([0,1])
    plt.savefig("main.png", dpi=600, bbox_inches='tight')
    plt.show()    


def mat_params():
    FN = np.loadtxt("false_negatives.csv")
    FP = np.loadtxt("false_positives.csv")
    TN = np.loadtxt("true_negatives.csv")
    TP = np.loadtxt("true_positives.csv")

    a = np.array([[np.mean(TP), np.mean(TN)], [np.mean(FP), np.mean(FN)]])
    sns.heatmap(a, annot=True, xticklabels=["Positive", "Negative"], yticklabels=["True", "False"])
    plt.savefig("cmat.png", dpi=600, bbox_inches='tight')
    plt.show()    

if __name__=='__main__':
    plot_params()
    mat_params()

