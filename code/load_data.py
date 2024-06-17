import csv
import torch
import random
from train import train
import numpy as np
import pandas as pd
  
    
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(cd_data)
    

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def dataset(args):
    dataset = dict()

    # dataset['c_d'] = read_csv(args.dataset_path + '/c_d.csv')
    dataset['M_D'] = read_csv('G:\\Code\\2\\dataset\\M_D.csv')
    

    zero_index = []
    one_index = []
    for i in range(dataset['M_D'].size(0)):
        for j in range(dataset['M_D'].size(1)):
            if dataset['M_D'][i][j] < 1:
                zero_index.append([i, j, 0])
            if dataset['M_D'][i][j] >= 1:
                one_index.append([i, j, 1])
   
    cd_pairs = random.sample(zero_index, len(one_index)) + one_index

    dd_matrix = read_csv('G:\\Code\\2\\dataset\\ID.csv')
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['DD'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}
    cc_matrix = read_csv('G:\\Code\\2\\dataset\\IM.csv')
    cc_edge_index = get_edge_index(cc_matrix)
    dataset['MM'] = {'data_matrix': cc_matrix, 'edges': cc_edge_index}

    return dataset, cd_pairs


def feature_representation(model, args, dataset):
    model.cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model = train(model, dataset, optimizer, args)
    model.eval()
    with torch.no_grad():
        score, miRNA_fea, dis_fea = model(dataset)
    miRNA_fea = miRNA_fea.cpu().detach().numpy()
    dis_fea = dis_fea.cpu().detach().numpy()
    return score, miRNA_fea, dis_fea


def new_dataset(mi_fea, dis_fea, cd_pairs):
    unknown_pairs = []
    known_pairs = []
    
    for pair in cd_pairs:
        if pair[2] == 1:
            known_pairs.append(pair[:2])
            
        if pair[2] == 0:
            unknown_pairs.append(pair[:2])
    
    nega_list = []
    for i in range(len(unknown_pairs)):
        nega = mi_fea[unknown_pairs[i][0],:].tolist() + dis_fea[unknown_pairs[i][1],:].tolist()+[0,1]
        nega_list.append(nega)
        
    posi_list = []
    for j in range(len(known_pairs)):
        posi = mi_fea[known_pairs[j][0],:].tolist() + dis_fea[known_pairs[j][1],:].tolist()+[1,0]
        posi_list.append(posi)
    
    samples = posi_list + nega_list
    
    random.shuffle(samples)
    samples = np.array(samples)
    return samples



def C_Dmatix(cd_pairs,trainindex,testindex):
    c_dmatix = np.zeros((788,374))  # (585,88)
    for i in trainindex:
        if cd_pairs[i][2]==1:
            c_dmatix[cd_pairs[i][0]][cd_pairs[i][1]]=1
    
    
    dataset = dict()
    cd_data = []
    cd_data += [[float(i) for i in row] for row in c_dmatix]
    cd_data = torch.Tensor(cd_data)
    dataset['M_D'] = cd_data
    
    train_cd_pairs = []
    test_cd_pairs = []
    for m in trainindex:
        train_cd_pairs.append(cd_pairs[m])
    
    for n in testindex:
        test_cd_pairs.append(cd_pairs[n])



    return dataset['M_D'], train_cd_pairs, test_cd_pairs


def updating_U(W, A, U, V, lam):
    m, n = U.shape
    fenzi = (W * A).dot(V.T)
    fenmu = (W * (U.dot(V))).dot((V.T)) + (lam / 2) * (np.ones([m, n]))
    U_new = U
    for i in range(m):
        for j in range(n):
            U_new[i, j] = U[i, j] * (fenzi[i, j] / fenmu[i, j])
    return U_new


def updating_V(W, A, U, V, lam):
    m, n = V.shape
    fenzi = (U.T).dot(W * A)
    fenmu = (U.T).dot(W * (U.dot(V))) + (lam / 2) * (np.ones([m, n]))
    V_new = V
    for i in range(m):
        for j in range(n):
            V_new[i, j] = V[i, j] * (fenzi[i, j] / fenmu[i, j])
    return V_new

def get_low_feature(k,lam, A):
    m, n = A.shape
    arr1=np.random.randint(0,100,size=(m,k))
    U = arr1/100#miRNA
    arr2=np.random.randint(0,100,size=(k,n))
    V = arr2/100#disease
    i = 0
    while i < 2500:
        i = i + 1
        U = updating_U(A, A, U, V, lam)
        V = updating_V(A, A, U, V, lam)
    return U, V.transpose()


def low_feature():
    low_dim_features = {}
    A = np.load('G:\\Code\\2\\dataset\\miRNA-disease association.npy')
    print('start Get low-dimensional features U and V')
    U, V = get_low_feature(64, 0.01, A)
    low_dim_features['U'] = U
    low_dim_features['V'] = V
    mirna_feature = low_dim_features['U']
    dis_feature = low_dim_features['V']
    print("dis_feature:", dis_feature.shape)
    print("mirna_feature:", mirna_feature.shape)
    print('finished Get low-dimensional features')

    return mirna_feature, dis_feature