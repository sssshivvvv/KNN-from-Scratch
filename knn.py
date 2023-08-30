from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from math import sqrt
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KNN:

    def __init__(self, dataset, k, dist_metric, encoder_type):
        
        self.dataset = dataset
        self.k = k
        self.dist_metric = dist_metric
        self.encoder_type = encoder_type

        #Loading Data
        self.data = np.load(self.dataset, allow_pickle=True)

        #Encodings
        self.Res_Code = self.data[:,1]
        self.vit_Code = self.data[:,2]

        #Labels
        self.Labels = self.data[:,3]
        
        #For performance measure
        X = self.data[:,0:3]
        Y = self.data[:,3]
        X_train, X_val, y_train, y_val = train_test_split(X,Y, test_size = 0.25, random_state=42)
        self.Res_code_train = X_train[:,1]
        self.vit_code_train = X_train[:,2]
        self.label_train = y_train

        
    def data_representation(self): #For first Image
        print(f"Image_ID {self.data[0,0]} \n\n")
        print(f"Resnet_Embedding {self.data[0,1]} \n\n")
        print(f"ViT_embedding {self.data[0,2]} \n\n")
        print(f"Label: {self.data[0,3]} \n\n")
        print(f"Guess_Time {self.data[0,4]} \n\n")


    def compute_distance(self, vect1, vect2):

        if self.dist_metric == 'Euclidean':
            return np.linalg.norm(vect1-vect2)

        elif self.dist_metric == 'Manhattan':
            return np.sum(np.abs(vect1-vect2))

        elif self.dist_metric == 'Cosine':
            return cosine(vect1.flatten(), vect2.flatten())

           
           
       
    def prediction(self, test):

        # Computing distance with training dataset.

        if self.encoder_type == 'Resnet':
            distances = []
            for i in range(self.Res_code_train.T.shape[0]):
                distances.append((self.label_train[i], self.compute_distance(self.Res_code_train[i], test)))
            
        
        elif self.encoder_type == 'ViT':
            distances = []
            for i in range(self.vit_code_train.T.shape[0]):
                distances.append((self.label_train[i], self.compute_distance(self.vit_code_train[i], test)))
            
        
        self.distances = sorted(distances, key=lambda x: x[1])
    

        # Getting nearest neighbours

        dist = self.distances
        only_labels = []


        for i in range(len(dist)):
            only_labels.append(dist[i][0])

        self.neighbours = []

        for i in range(self.k):
            self.neighbours.append(dist[i])

        self.neighbour_labels = []
        
        for i in range(self.k):
            self.neighbour_labels.append(only_labels[i])
           
       
       # prediction when all the labels in the neighbours are unique. 

        if len(self.neighbours)==len(set(self.neighbour_labels)):
            min_label = None
            min_num = float('inf')
            for label, dist in self.neighbours:
                if dist<min_num:
                    min_label = label 
                    min_num = dist
            
            self.predicted_label = min_label
            
            return min_label
            
        
       # prediction hen the neighbours are not unique.
        elif len(self.neighbours)!=len(set(self.neighbour_labels)):

            y = self.neighbours
            z = list(set(self.neighbour_labels))

            count = 0
            label_count = []

            for i in range(len(z)):
                for j in range(len(y)):
                    if z[i]==y[j][0]:
                       count+=1
                label_count.append((z[i],count))
                count = 0
                    
    
            mode, counts = max(label_count, key=lambda x: x[1])
            self.predicted_label = mode
            
            return mode


    def performance_measure(self,test_file):  

        test_data = np.load(test_file, allow_pickle=True)
        Res_code_test = test_data[:,0]
        vit_code_test = test_data[:,1]
        gt_label = test_data[:,2]
        
        self.predicted = []
        self.ground_truth = []
        self.accuracy = 0

        if self.encoder_type=='Resnet':
            for i in range(len(gt_label)):
                self.predicted.append(self.prediction(Res_code_test[i]))
                self.ground_truth.append(gt_label[i])

            self.accuracy = accuracy_score(self.ground_truth, self.predicted )
            self.precision = precision_score(self.ground_truth, self.predicted, average ='weighted', zero_division = 1)
            self.recall = recall_score(self.ground_truth, self.predicted, average ='weighted', zero_division = 1)
            self.f1 = f1_score(self.ground_truth, self.predicted, average ='weighted', zero_division = 1)


        
        if self.encoder_type=='ViT':
            for i in range(len(gt_label)):
                self.predicted.append(self.prediction(vit_code_test[i]))
                self.ground_truth.append(gt_label[i])


            self.accuracy = accuracy_score(self.ground_truth, self.predicted )
            self.precision = precision_score(self.ground_truth, self.predicted, average ='weighted', zero_division = 1)
            self.recall = recall_score(self.ground_truth, self.predicted, average ='weighted', zero_division = 1)
            self.f1 = f1_score(self.ground_truth, self.predicted, average ='weighted', zero_division = 1)
           
         


#Testing
# a = KNN('data.npy', 5 ,'Cosine', 'Resnet')
# a.performance_measure()
# print(a.accuracy*100)


#type 

#$ python eval.sh test_data.npy  \\In terminal to run the bash script.
