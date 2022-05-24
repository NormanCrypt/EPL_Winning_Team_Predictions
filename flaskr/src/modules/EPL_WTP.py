from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,roc_auc_score
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import plotly.express as px

class EPL_WinningTeamPrediction(object):
    def __init__(self,train_data,test_data):
        self.train_dataset = train_data
        self.test_dataset = test_data
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def process_data(self):
        
        useful_data = self.train_dataset.copy()
        useful_test_data = self.test_dataset.copy()

        useful_data = useful_data.drop(["Div","Date"],axis=1)
        useful_test_data = useful_test_data.drop(["Div","Date"],axis=1)
        
        useful_data = useful_data.drop(["HTR","Referee"],1)
        useful_data["Result"] = useful_data.apply(lambda row: self.transform_data(row),axis=1)

        useful_test_data = useful_test_data.drop(["HTR","Referee"],1)
        useful_test_data["Result"] = useful_test_data.apply(lambda row: self.transform_data(row),axis=1)
        
        self.X_train = useful_data[['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','HF','AF','HY','AY','HR','AR','HC','AC','AST']]
        self.y_train = useful_data['Result']

        self.X_test = useful_test_data[['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','HF','AF','HY','AY','HR','AR','HC','AC','AST']]
        self.y_test = useful_test_data['Result']
        
        return[self.X_train,self.X_test,self.y_train,self.y_test]
    
    def transform_data(self,record):
        
        if(record.FTR == 'H'):
            return 1
        elif(record.FTR == 'A'):
            return -1
        else:
            return 0 
        
    def SVM_Model(self,kernel='linear',probability=True):
        
        clf = SVC(kernel='poly',probability=True)

        y_pred = clf.fit(self.X_train,self.y_train).predict(self.X_train)
        score = accuracy_score(y_pred,self.y_train)

        svm_model = clf.fit(self.X_train,self.y_train)
        
        return[svm_model,score]
    
    def Random_Forest_Model(self):
        
        clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
        
        y_pred = clf.fit(self.X_train,self.y_train).predict(self.X_train)
        score = accuracy_score(y_pred,self.y_train)

        rf_model = clf.fit(self.X_train,self.y_train)
        
        return[rf_model,score]
    
    def Descision_Tree_Model(self):
        
        clf = DecisionTreeClassifier(max_depth=5)
        
        y_pred = clf.fit(self.X_train,self.y_train).predict(self.X_train)
        score = accuracy_score(y_pred,self.y_train)

        dt_model = clf.fit(self.X_train,self.y_train)
        
        return[dt_model,score]
    
    def KNeighbors_Model(self):
        
        clf = KNeighborsClassifier(n_neighbors = 10)
        
        y_pred = clf.fit(self.X_train,self.y_train).predict(self.X_train)
        score = accuracy_score(y_pred,self.y_train)

        knn_model = clf.fit(self.X_train,self.y_train)
        
        return[knn_model,score]
    
    def trainModels(self):
        
        svm_model,svm_score = self.SVM_Model()
        rf_model,rf_score = self.Random_Forest_Model()
        dt_model,dt_score = self.Descision_Tree_Model()
        knn_model,knn_score = self.KNeighbors_Model()
        
        return[svm_model,svm_score,rf_model,rf_score,dt_model,dt_score,knn_model,knn_score]
    
    def predict(self,data,model):
    
        prediction = model.predict(data)
        return prediction
    
    def predict_proba(self,data,model):
        
        prediction_probabilities = model.predict_proba(data)
        return prediction_probabilities
    
    def evaluate(self,model_data):
        
        model = model_data[0]
        score = model_data[1]
        
        y_test_binarized=label_binarize(self.y_test[:150],classes=np.unique(self.y_test))
        
        fig = go.Figure()
        fig.add_shape(
            type="line" , line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        pred_proba = self.predict_proba(self.X_test[:150],model)
        
        for i in range(pred_proba.shape[1]):
            y_true = y_test_binarized[:, i]
            y_score = pred_proba[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_test_binarized[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="y", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=800, height=650
        )
                
        return[fig,pred_proba,fpr,tpr,score]
    
    def evaluateModels(self):
        return