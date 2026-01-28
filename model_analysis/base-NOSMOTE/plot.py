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

    fig, ax = plt.subplots(2, 2, figsize=[12,8])

    def plot_lines(axes, var, var_name, color, title=None):
        axes.plot(var, label=var_name+f" {np.mean(var):.2f}$\pm${np.std(var):.2f}", color=color, lw=2)
        axes.scatter(range(100), var, color=color, lw=2)
        axes.set_title(var_name)
        if title:
            axes.set_title(title)
        
        axes.legend(loc="upper right")
        axes.set_xlim([0,100])
        axes.set_ylim([0,1])

    plot_lines(ax[0,0], accuracy, "Accuracy", "red")
    plot_lines(ax[1,1], auc, "AUC", "darkorchid")
    plot_lines(ax[1,0], fscore, "F1 score", "maroon")
    plot_lines(ax[0,1], precision, "Precision", "blue")
    plot_lines(ax[0,1], recall, "Recall", "dodgerblue", title="Recall and Precision")
    
    for ax in ax.flat:
        ax.set(xlabel="Iteration", ylabel="Score")
        ax.label_outer()
    
    #plt.xlabel("Iteration", size=14)
    #plt.ylabel("Score", size=14)
    #plt.xticks(size=12)
    #plt.yticks(size=12)
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

