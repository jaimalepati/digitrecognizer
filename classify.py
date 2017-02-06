"""
Program to classify the data into classes based on KNN classification
In this program, we have used MNIST Database with 60000 training images and 10000 test images

Steps Followed
1. Initially the data from the MNIST database has been converted into CSV files
2. A sample of 6000 training images and 1000 test images has been selected
3. We perfomed the KNN classification on the images and classified them into classes
4. We tried to calculate the accuracy of our classification
"""

import numpy as np
import time
import scipy.spatial.distance as ssd

def read_data(fn):                         #Reading the data from the csv files and storing into data and label matrices
    with open(fn) as f:
        raw_data = np.loadtxt(f, delimiter= ',', dtype="float",
            skiprows=1, usecols=None)
    data = []; label = []

    for row in raw_data:
        data.append(row[1:-1])
        label.append(int(row[:1]))

    return np.array(data), np.array(label)


def knn(k, dtrain, dtest, dtr_label, dist):      #Running the knn classification function
    pred_class = []
    for i, di in enumerate(dtest):
        distances = []
        for j, dj in enumerate(dtrain):
            distances.append((calc_dist(di,dj,dist), j))    #Calculating the distances between the images

        k_nn = sorted(distances)[:k]                        #Sorting the distances in ascending order
        pred_class.append(classify(k_nn, dtr_label))        #Predicting the label of the test images

    return pred_class


def binarise(d):   #Binarising the points for the case of Hamming Distance
    for d in range(255):
        d = d/255
    return d

def calc_dist(di,dj,i):    #Distance Calculation and Different types of functions
    if i == 1:
        return ssd.euclidean(di,dj) # built-in Euclidean fn
    elif i == 2:
        binarise(di)
        binarise(dj)
        return ssd.hamming(di,dj)   # built-in Hamming fn - Calculates Normalised Hamming distance
    elif i == 3:
        return ssd.cosine(di,dj)    # built-in Cosine fn
    elif i == 4:
        return ssd.cityblock(di,dj) # built-in L1 distance
    elif i == 5:
        return  ssd.minkowski(di,dj,3)  #built-in L3 distance

def classify(k_nn, dtr_label):              #Function to retrieve the label from the classification
    label = []
    for dist, idx in k_nn:
        label.append(dtr_label[idx])
    return np.argmax(np.bincount(label))    #Returning the label which appears maximum number of times

def accuracy(true_class, pred_class):           #Calculating the accuracy of the classification
	correct = 0
	for x in range(len(true_class)):
		if true_class[x] == pred_class[x]:
			correct += 1
	return (correct/float(len(true_class))) * 100.0


def main():
    """ k-nearest neighbors classifier """

    # initialize runtime
    start = time.clock()

    dtrain, dtr_label = read_data('mnist_train.csv')
    dtest, true_class = read_data('mnist_test.csv')

    #For a sample of images, as the time taken for all the images is huge
    dtrain = dtrain[1:6000]
    dtr_label = dtr_label[1:6000]
    dtest = dtest[1:1000]
    true_class = true_class[1:1000]

    # Initialize K with different values [1,3,5,7] or other if required
    K = 1
    dist_fn = 1 # For different type of distances different values [2,3,4,5]
    results = []
    # predicting the data test into class
    pred_class = knn(K, dtrain, dtest, dtr_label, dist_fn)
    #Assesing the accuracy of the predicted class
    acc = accuracy(true_class,pred_class)

    # printing the accuracy of prediction
    print( "The accuracy of the prediction: ", acc )
    print("Error Performance: ", 100-acc)

    # retrieving the time
    run_time = time.clock() - start
    print("Runtime:", run_time)

if __name__ == '__main__':
    main()
