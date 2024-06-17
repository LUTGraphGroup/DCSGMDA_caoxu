from param import parameter_parser
import load_data
from model import CDSG
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import time
from sklearn import metrics
from utils import plot_auc_curves, plot_prc_curves
import numpy as np

def CDA(n_fold):
    args = parameter_parser()
    dataset, M_D_pairs = load_data.dataset(args)

    kf = KFold(n_splits = n_fold, shuffle = True)
    model = CDSG(args)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []
    fprs = []
    tprs = []
    precisions = []
    recalls = []


    localtime = time.asctime( time.localtime(time.time()) )
    with open() as f:
        f.write('time:\t'+ str(localtime)+"\n")


        fold_count = 0
        for train_index, test_index in kf.split(M_D_pairs):
            fold_count += 1  # 折数加1
            print("--------------------")
            print('Training for Fold:', fold_count)  # 打印折数

            M_Dmatix,train_MD_pairs,test_MD_pairs = load_data.C_Dmatix(M_D_pairs,train_index,test_index)
            dataset['M_D']=M_Dmatix
            score, mi_fea, dis_fea = load_data.feature_representation(model, args, dataset)
            print("This is the fea_shape of miRNAs and diseases extracted from the training dataset")
            train_dataset = load_data.new_dataset(mi_fea, dis_fea, train_MD_pairs)
            print("This is the fea_shape of miRNAs and diseases extracted from the test dataset")
            test_dataset = load_data.new_dataset(mi_fea, dis_fea, test_MD_pairs)

            X_train, y_train = train_dataset[:, :-2], train_dataset[:, -2:]
            X_test, y_test = test_dataset[:,:-2], test_dataset[:,-2:]


            clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                                learning_rate_init=0.0001, max_iter=1000)
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)
            y_prob = y_prob[:, 0]


            fpr, tpr, thresholds = metrics.roc_curve(y_test[:,0], y_prob)

            precision, recall, _ = metrics.precision_recall_curve(y_test[:,0], y_prob)
            test_auc = metrics.auc(fpr, tpr)
            test_prc = metrics.auc(recall, precision)


            pred_test = [0 if j < 0.5 else 1 for j in y_prob]
            acc_test = metrics.accuracy_score(y_test[:,0], pred_test)
            pre_test = metrics.precision_score(y_test[:,0], pred_test)
            recall_test = metrics.recall_score(y_test[:,0], pred_test)
            f1_test = metrics.f1_score(y_test[:,0], pred_test)

            print('Fold: ', fold_count, 'Test acc: %.4f' % acc_test, 'Test Pre: %.4f' % pre_test,
                  'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test PRC: %.4f' % test_prc,
                  'Test AUC: %.4f' % test_auc)

            auc_result.append(test_auc)
            acc_result.append(acc_test)
            pre_result.append(pre_test)
            recall_result.append(recall_test)
            f1_result.append(f1_test)
            prc_result.append(test_prc)

            fprs.append(fpr)
            tprs.append(tpr)
            precisions.append(precision)
            recalls.append(recall)

        print('## Training Finished !')
        print('---------------------------------------------------00--------------------------------------------')
        print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
              'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
              'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
              'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
              'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
              'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

        plot_auc_curves(fprs, tprs, auc_result, n_fold=n_fold, directory='G:\\Code\\2\\AUCAUPRC', name='DCSGMDA_auc')
        plot_prc_curves(precisions, recalls, prc_result, n_fold=n_fold, directory='G:\\Code\\2\\AUCAUPRC', name='DCSGMDA_prc')


if __name__ == "__main__":
    
    n_fold = 10
    for i in range(1):
        CDA(n_fold)
