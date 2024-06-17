import numpy as np
import matplotlib.pyplot as plt
from scipy import interp


def plot_auc_curves(fprs, tprs, auc, n_fold,directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.6, linestyle='--', label='ROC Fold {} (AUC = {:.4f})'.format(i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='black', alpha=0.9,
             label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ({}-fold Cross Validation)'.format(n_fold))
    plt.legend(loc='lower right')
    plt.legend(loc='lower left', fontsize='small')
    plt.tight_layout(pad=6.0)
    plt.savefig(directory+'/%s.png' % name, dpi=600, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, n_fold, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.6, linestyle='--', label='PR Fold {} (AUPRC = {:.4f})'.format(i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0.0
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='black', alpha=0.9,
             label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve ({}-fold Cross Validation)'.format(n_fold))
    plt.legend(loc='lower left')
    plt.legend(loc='lower left', fontsize='small')
    plt.tight_layout(pad=6.0)

    plt.savefig(directory + '/%s.png' % name, dpi=600, bbox_inches='tight')
    plt.close()