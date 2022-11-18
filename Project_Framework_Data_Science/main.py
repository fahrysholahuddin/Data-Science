from dataset import class_dataset
import numpy as np
from missingValue import class_missingValue
from classifier import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


import numpy as np
if __name__ == '__main__':
    #load dataset
    # glcm : all fitur
    # dissimilarity : dis fitur
    data = class_dataset(dataset_name="glcm")
    print("Jumlah data keseluruhan:", len(data.y))
    
    #missing value
    jumlah_na = np.isnan(data.X).sum()
    print("jumlah missing value pada data:", jumlah_na)
    if jumlah_na != 0:
        data = class_missingValue(data, pick="mean")
    
    #preprocesing (normalisasi)
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.33, random_state=42)

    ## model KNN
    knn = class_KNN(X_train, y_train)
    knn.model()
    knn_y_pred = knn.predict(X_test)

    ## model NB
    nb = class_NB(X_train, y_train)
    nb.model()
    nb_y_pred = nb.predict(X_test)
    
    #evaluasi
    #Accuracy = TP / Jumlah Data
    #precision = TP / (TP + FP)
    #recall = TP / (TP + FN)
    #f1-Score = 2*(precision*recall)/(precision+recall)
    
    print("Jumlah data testing:", len(y_test))
    
    nb_acc = accuracy_score(y_test, nb_y_pred)
    print("accuracy NB: ",round(nb_acc,2))
    nb_cm = confusion_matrix(y_test, nb_y_pred)
    print("confusion matrix untuk model NB: \n",nb_cm)
    print(classification_report(y_test, nb_y_pred))
    
    knn_acc = accuracy_score(y_test, knn_y_pred)
    print("accuracy KNN: ",round(knn_acc,2))
    knn_cm = confusion_matrix(y_test, knn_y_pred)
    print("confusion matrix untuk model KNN: \n",knn_cm)
    print(classification_report(y_test, knn_y_pred))