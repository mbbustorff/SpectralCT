# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:32:21 2021

@author: mbust
"""
from AttenuationDB_simple import *

import itertools
import sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sn

from ordered_set import OrderedSet

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier




#--------------------setup the data-----------------------
#-labels
#-df type file
datapath_head1 = "../Sample_06062018_Fluids"
file1 = datapath_head1 + "/processed/segmented/"+'labels_all.txt'
datapath_head2 = "../Sample_06062018_NonThreat_74proj"
file2 = datapath_head2 + "/processed/segmented/"+'labels_all.txt'
datapath_head3 = "../Sample_06062018_Threat_74proj"
file3 = datapath_head3 + "/processed/segmented/"+'labels_all.txt'
datapath_head4 = "../Sample_23012018"
file4 = datapath_head4 + "/processed/segmented/"+'labels_all.txt'
datapath_head5 = "../Sample_24012018"
file5 = datapath_head5 + "/processed/segmented/"+'labels_all.txt'

all_datapath_heads = [datapath_head1,datapath_head2,datapath_head3,datapath_head4,datapath_head5]
all_files = [file1,file2,file3,file4,file5]
all_filename=["06062018_Fluids","NonThreat_74proj","Threat_74proj","Sample_23012018","Sample_24012018"]

all_label_names = [[],[],[],[],[]]
all_label_names_set = [[],[],[],[],[]]
all_label_ids = [[],[],[],[],[]]
all_label_ids_set = [[],[],[],[],[]]

labels1 = ["acetone","h2o","h2o2","nitric_acid","olive_oil","whiskey"]
labels2 = ["acetone","c4","h2o2","methanol"]
labels3 = ["nivea", "olive_oil", "toothpaste","h2o","cien"]
labels4 = ["h2o", "h2o2", "whiskey","hand_cream","toothpaste","c4","aluminium"]
labels5 = ["aluminium", "c4", "hand_cream","h2o2","toothpaste","h2o","whiskey"]
labels = labels1+labels2+labels3+labels4+labels5
labels = list(OrderedSet(labels))

#-------------Setup labels & ids ---------------------
for v in range(len(all_files)):
    file = all_files[v]
    filename = all_filename[v]
    print("Seting-up file: " + filename)
    
    
    labels_all = pd.read_csv(file, delimiter = "\t", header=None)
    flat_labels = labels_all.to_numpy().flatten()
    label_names_set = list(OrderedSet(flat_labels))
    label_ids = [labels.index(x) for x in flat_labels]
    label_ids_set = list(OrderedSet(label_ids))
    
    print(label_names_set)
    print(label_ids_set)
    print()
    
    all_label_names[v] = flat_labels 
    all_label_names_set[v] = label_names_set
    all_label_ids[v] = label_ids
    all_label_ids_set[v] = label_ids_set
        
all_data_all = [[],[],[],[],[]]

#------------Load LAVs data-------------------------------
for v in range(len(all_files)):
    file = all_files[v]
    filename = all_filename[v]
    print("Loading data for file: "+filename)

    data_file = all_datapath_heads[v] + "/processed/segmented/"+'LAC_all.csv'
        
    columns = list(np.arange(1,129))

    data_all = pd.read_csv(data_file, header=None)
    data_all.columns = columns
    data_all['ids'] = all_label_ids[v]
    
    all_data_all[v] = data_all
    
data_master = pd.concat([all_data_all[0], all_data_all[1], all_data_all[2], all_data_all[3], all_data_all[4]], axis=0)

print("Data master created (combination of all dataframes), here's the distribution of each label:")
for i in range(13):
    print(str(i)+" ({:}): ".format(labels[i])+str(len(data_master.ids[data_master.ids==i]))+" voxels")

#data_master has all values, last column is segment
#-------------setup the data - done -----------------------

import matplotlib.pyplot as plt
import numpy as np

y = [len(data_master.ids[data_master.ids==i]) for i in range(13)]
mylabels = labels

plt.pie(y, labels = mylabels)
plt.title("Voxels distribution")
plt.savefig("piechart.pdf",dpi = 200)
plt.show() 


#split train and test data
y=all_label_ids[0]+all_label_ids[1]+all_label_ids[2]+all_label_ids[3]+all_label_ids[4]
X, y = data_master.iloc[:,:-1],y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    

def GNB_train_test():
    print("Creating sklearning model...")
    
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    
    plt.figure(figsize = (10,7))
    plot_confusion_matrix(gnb, X_test, y_test, display_labels=labels,normalize='true') 
    plt.xticks(range(12), labels, rotation='vertical')
    plt.title("Confusion Matrix for GNB (normalized)")
    plt.savefig("confusion_gaussNB_masterNormal.pdf")
    
    plt.figure(figsize = (10,7))
    plot_confusion_matrix(gnb, X_test, y_test, display_labels=labels) 
    plt.xticks(range(12), labels, rotation='vertical')
    plt.title("Confusion Matrix for GNB")
    plt.savefig("confusion_gaussNB_master.pdf",dpi = 200)
    
    plt.show()
    
    print("Jaccard Coefficient:")
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    jaccard = sklearn.metrics.jaccard_score(y_test, y_pred, labels=None, pos_label=1, average=None, sample_weight=None)
    
    jaccard_table = pd.DataFrame(np.vstack((np.arange(12),jaccard)))
    
    print("Overall score = {:}".format(accuracy))
    #print("Jaccard coefficient for each label:")
    #print(jaccard_table)

def KNN_train_test():
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    plt.figure(figsize = (10,7))
    plot_confusion_matrix(knn, X_test, y_test, display_labels=labels,normalize='true') 
    plt.xticks(range(12), labels, rotation='vertical')
    plt.title("Confusion Matrix for KNN (normalized)")
    plt.savefig("confusion_KNN_masterNormal.pdf")
    
    plt.figure(figsize = (10,7))
    plot_confusion_matrix(knn, X_test, y_test, display_labels=labels) 
    plt.xticks(range(12), labels, rotation='vertical')
    plt.title("Confusion Matrix for KNN")
    plt.savefig("confusion_KNN_master.pdf",dpi = 200)
    plt.show()
    
    print("Jaccard Coefficient:")
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    jaccard = sklearn.metrics.jaccard_score(y_test, y_pred, labels=None, pos_label=1, average=None, sample_weight=None)
    
    jaccard_table = pd.DataFrame(np.vstack((np.arange(12),jaccard)))
    
    print("Overall score = {:}".format(accuracy))
    #print("Jaccard coefficient for each label:")
    #print(jaccard_table)


#--------------create DB and references---------------
DB = AttenuationDB()


acetone_ref1 = DB._db["acetone"][0]
h2o_ref1 = DB._db["h2o"][0]
h2o2_ref1 = DB._db["h2o2_30p"][0]
nitric_acid_ref1 = DB._db["nitric_acid"][0]
olive_oil_ref1 = DB._db["olive_oil"][0]
whiskey_ref1 = DB._db["whiskey"][0]
c4_ref1=DB._db["c4_simulant"][0]
methanol_ref1=DB._db["methanol"][0]


ref1 = np.array([acetone_ref1,h2o_ref1,h2o2_ref1,nitric_acid_ref1,olive_oil_ref1,whiskey_ref1,c4_ref1,methanol_ref1])

#return averaged_LAC_128,averaged_LAC_32,ref1

#--------------create DB and references - done---------------


def reference_individual(data_master):
    
    data_master = data_master.to_numpy()
    newData0 = data_master[data_master[:,-1]==0]
    newData1 = data_master[data_master[:,-1]==1]
    newData2 = data_master[data_master[:,-1]==2]
    newData3 = data_master[data_master[:,-1]==3]
    newData4 = data_master[data_master[:,-1]==4]
    newData5 = data_master[data_master[:,-1]==5]
    newData6 = data_master[data_master[:,-1]==6]
    newData7 = data_master[data_master[:,-1]==7]
    
    newData = np.vstack((newData0,newData1,newData2,newData3,newData4,newData5,newData6,newData7))
    
    newData_32 = np.array([0]*len(newData))
    
    i=0
    while i<=128:
        column = newData[:,i:i+4].mean(axis=1)
        newData_32 = np.vstack((newData_32,column))
        i+=4
    newData_32 = newData_32.T[:,1:]
    
    norm_matrix1 = np.array([[0.0]*8]*len(newData_32))
    norm_matrix2 = np.array([[0.0]*8]*len(newData_32))
    for i in range(len(newData_32)):
        for j in range(8):
            dif = newData_32[i][:-1]-ref1[j]
            
            norm1 = np.linalg.norm(dif, ord=1)
            norm_matrix1[i][j] = norm1
            
            norm2 = np.linalg.norm(dif)
            norm_matrix2[i][j] = norm2
            
    index_vec = [0]*len(newData)
    closest_ref = [0]*len(newData)
    
    norm_matrix = norm_matrix1
    for i in range (len(newData)):
        index_vec[i] = list(norm_matrix[i]).index(min(norm_matrix[i]))
        #closest_ref[i] = (labels[index_vec[i]],labels[i])
        #print("Closest: {}. Actual label: {}".format(labels[index_vec[i]],labels[i]))
    
    index_vec = np.vstack((index_vec,newData_32[:,-1]))
    tru = index_vec[0]
    pred = index_vec[1]
    
    conf_matrix = confusion_matrix(tru,pred)
    
    
    df_cm = pd.DataFrame(conf_matrix, index = [i for i in labels[0:-5]],
                      columns = [i for i in labels[0:-5]])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.xticks(np.arange(8)+0.5, labels[0:-5], rotation='vertical')
    plt.yticks(np.arange(8)+0.5, labels[0:-5])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix using L2-norm with the reference (Individual voxels)")
    plt.savefig("confusion_referenceIndividualVox_master.pdf",dpi = 200)
    plt.show()
    
    print("Jaccard Coefficient:")
    accuracy = sklearn.metrics.accuracy_score(tru, pred, normalize=True, sample_weight=None)
    jaccard = sklearn.metrics.jaccard_score(tru, pred, labels=None, pos_label=1, average=None, sample_weight=None)
    
    jaccard_table = pd.DataFrame(np.vstack((np.arange(8),jaccard)))
    
    print("Overall score = {:}".format(accuracy))
    #print("Jaccard coefficient for each label:")
    #print(jaccard_table)
    
    
    tru = [0,1,2,3,4,5,6,7]
    pred = [0,0,0,0,0,0,0,0]
    for i in range(8):
        pred[i] = list(conf_matrix[i]).index(max(list(conf_matrix[i])))
    conf_matrix = confusion_matrix(tru,pred)
    
    df_cm = pd.DataFrame(conf_matrix, index = [i for i in labels[0:-5]],
                      columns = [i for i in labels[0:-5]])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.xticks(np.arange(8)+0.5, labels[0:-5], rotation='vertical')
    plt.yticks(np.arange(8)+0.5, labels[0:-5])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix using L2-norm with the reference (Majority vote assignment)")
    plt.savefig("confusion_referenceIndividualMaj_master.pdf",dpi = 200)
    plt.show()
    
    print("Jaccard Coefficient:")
    accuracy = sklearn.metrics.accuracy_score(tru, pred, normalize=True, sample_weight=None)
    jaccard = sklearn.metrics.jaccard_score(tru, pred, labels=None, pos_label=1, average=None, sample_weight=None)
    
    jaccard_table = pd.DataFrame(np.vstack((np.arange(8),jaccard)))
    
    print("Overall score = {:}".format(accuracy))
    #print("Jaccard coefficient for each label:")
    #print(jaccard_table)
    
    
    return index_vec,conf_matrix
            

def reference_average(data_master):
    #-----------get the average LACs --------------------------
    #- take the mean for all voxels of a segment
    #- then combine the 128 spectral channels into 32

    #data_master = data_master[data_master!=0]
    means = np.zeros(len(data_master.iloc[0]))
    for v in range(len(labels)):
        row = data_master[data_master['ids']==v].mean().to_numpy().T
        means = np.vstack((means,row))
    means = means[1:-5]
    
    averaged_LAC_128 = means
    averaged_LAC_32 = np.array([0]*len(means))
    
    
    #for i in range(len(means)):
    i=0
    while i<=128:
        column = averaged_LAC_128[:,i:i+4].mean(axis=1)
        averaged_LAC_32 = np.vstack((averaged_LAC_32,column))
        i+=4
    averaged_LAC_32 = averaged_LAC_32.T[:,1:]
    
    
    
    #-------------get the average LACs - done ----------------
    
    #---------------compute closest norms between LACs and ref----------- 
    norm_matrix1 = np.array([[0.0]*8]*8)
    norm_matrix11 = np.array([[0.0]*8]*8)
    norm_matrix2 = np.array([[0.0]*8]*8)
    for i in range(8):
        for j in range(8):
            dif = averaged_LAC_32[i][:-1]-ref1[j]
            norm1 = np.linalg.norm(dif)
            norm_matrix1[i][j] = norm1
            
            norm1 = np.linalg.norm(dif, ord=1)
            norm_matrix11[i][j] = norm1
            
            norm2=np.sum(np.power((averaged_LAC_32[i][:-1]-ref1[j]),2))
            norm_matrix2[i][j] = norm2
            
    norm_matrix = norm_matrix2
    index_vec = [0,0,0,0,0,0,0,0]
    closest_ref = [0,0,0,0,0,0,0,0]
    for i in range (8):
        index_vec[i] = list(norm_matrix[i]).index(min(norm_matrix[i]))
        closest_ref[i] = (labels[index_vec[i]],labels[i])
        #print("Closest: {}. Actual label: {}".format(labels[index_vec[i]],labels[i]))
    
    tru = [0,1,2,3,4,5,6,7]
    pred = index_vec
    
    conf_matrix = confusion_matrix(tru,pred)
    
    df_cm = pd.DataFrame(conf_matrix, index = [i for i in labels[0:-5]],
                      columns = [i for i in labels[0:-5]])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.xticks(np.arange(8)+0.5, labels[0:-5], rotation='vertical')
    plt.yticks(np.arange(8)+0.5, labels[0:-5])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix using L2-norm with the reference (Average segements LAC)")
    plt.savefig("confusion_referenceAverage_master.pdf",dpi = 200)
    plt.show()
    
    print("Jaccard Coefficient:")
    accuracy = sklearn.metrics.accuracy_score(tru, pred, normalize=True, sample_weight=None)
    jaccard = sklearn.metrics.jaccard_score(tru, pred, labels=None, pos_label=1, average=None, sample_weight=None)
    
    jaccard_table = pd.DataFrame(np.vstack((np.arange(8),jaccard)))
    
    print("Overall score = {:}".format(accuracy))
    #print("Jaccard coefficient for each label:")
    #print(jaccard_table)
    

'''
print("-                   -")
print("---------------------")
print("-                   -")
print("---------------------")
print("-                   -")
print("GNB")
GNB_train_test()
print("-                   -")
print("---------------------")
print("-                   -")
print("---------------------")
print("-                   -")
print("KNN")
KNN_train_test()
'''
print("-                   -")
print("---------------------")
print("-                   -")
print("---------------------")
print("-                   -")
print("Reference individual x2")
reference_individual(data_master)
print("-                   -")
print("---------------------")
print("-                   -")
print("---------------------")
print("-                   -")
print("Reference average")
reference_average(data_master)
print("                   ")
print("                   ")
print("                   ")
