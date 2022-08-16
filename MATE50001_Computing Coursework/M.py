

# In[1]:


get_ipython().run_line_magic('matplotlib', 'nbagg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.widgets
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_score
from sklearn.metrics import f1_score
from pandas.core.frame import DataFrame


# In[2]:


class Model():
    """A model - declared using KNN, Logistic Regression and SVC"""
    def __init__(self,
    file: str,
    y_col: str
    ):
        self.file = file
        self.origin_col = None
        self.data = None
        self.x_col = None
        self.y_col = y_col
        self.x_data = None
        self.y_data = None
        
        # n_neighbors parameter for knn
        self.range_k = [i for i in range(2, 6)]
        
        # C parameter for logistic regression
        self.range_l = [0.5, 1, 1.5]
        
        # kernels parameter for SVC
        self.range_s = ['linear', 'poly', 'rbf']
        
    # build a label encoder    
    def label_encoder(self, classes: list, variable: str):
        values = {
            classes[i] : i for i in range(len(classes))
            }
        self.data[variable] = self.data[variable].apply(lambda x : values[x])
    
    # preprocessing the data
    def get_data(self):
        self.data = pd.read_csv(self.file)
        self.origin_col = list(self.data.columns)
        
        # remove N/A value
        self.data = self.data.dropna()
        
        # remove 'Other' in column['gender']
        self.data = self.data.drop(self.data[self.data['gender'] == 'Other'].index)
        self.data = self.data.reset_index(drop=True)
        
        # remove column['id'] 
        del self.data['id']
        
        # label encoder of data['smoking status'] and label with [0,1,2,3]
        self.label_encoder([
            'Unknown', 
            'never smoked', 
            'formerly smoked', 
            'smokes'], 
            'smoking_status'
            )
        
        # one-hot encoder
        self.data = pd.get_dummies(self.data)
        
        # min-max
        columns = list(self.data.columns)
        data_minmax = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(self.data), columns=columns)
        for i in columns:
            self.data[i] = data_minmax[i]
        
        # split the data into different columns
        columns.remove(self.y_col)
        self.x_col = columns
        self.x_data = self.data[self.x_col]
        self.y_data = self.data[self.y_col]
    
    # get the data after removing a selected variable
    def data_without_one_var(self, data: np.ndarray, var:str) -> pd.DataFrame:
        for each in self.x_col:
            if each.startswith(var):
                del data[each]
        return data
    
    # split the data into train data and test data
    def split_data(self, xdata: np.ndarray, ydata: np.ndarray) -> list:
        
        # train:test = 9:1, two classes of outcome
        skf = StratifiedKFold(n_splits=10)  
        for train, test in skf.split(xdata, ydata):
            x_train = xdata[train]
            y_train = ydata[train]
            x_test = xdata[test]
            y_test = ydata[test]
            break
        return x_train, x_test, y_train, y_test
    
    # cross-validation of K-nearest-neighbour
    def knn_cv(self, x_train: np.ndarray, y_train: np.ndarray) -> list:
        cv_scores = []
        for i in self.range_k:
            knn = KNeighborsClassifier(n_neighbors=i, weights='distance')

            # use data_train to do CV
            scores = cross_val_score(knn,
                                     x_train,
                                     y_train,
                                     cv=5,
                                     scoring='f1'
            ) 
            cv_scores.append(scores.mean())
        
        # return mean cross validation score for each K
        return cv_scores
    
    # cross-validation of Logistic Regression
    def lr_cv(self, x_train: np.ndarray, y_train: np.ndarray) -> list:
        cv_scores = []
        for i in self.range_l:
            lr = LogisticRegression(class_weight='balanced', C=i, max_iter=3000)
            scores = cross_val_score(lr,
                                     x_train,
                                     y_train,
                                     cv=5,
                                     scoring='f1'
            )
            cv_scores.append(scores.mean())
        return cv_scores
    
    # cross-validation of support-vector-machine
    def svc_cv(self, x_train: np.ndarray, y_train: np.ndarray) -> list:
        cv_scores = []
        for i in self.range_s:
            rf = svm.SVC(class_weight='balanced', kernel=i)
            scores = cross_val_score(rf,
                                     x_train,
                                     y_train,
                                     cv=5, 
                                     scoring='f1'
            )
            cv_scores.append(scores.mean())
        return cv_scores
    
    # cvf1 --> f1-score of cross validation, score --> f1-score of model 
    def get_cvf1_and_score(self, x_data: np.ndarray, y_data: np.ndarray, model: list) -> dict:
        cvf1 = {}
        score = {}
        xdata = x_data.to_numpy()
        ydata = y_data.to_numpy()
        x_train, x_test, y_train, y_test = self.split_data(xdata, ydata)
        
        for each in model:
            # use K-Nearest-neighbour classifier
            if each == 'KNeighborsClassifier':
                a = self.knn_cv(x_train, y_train)

                # K begin with 2 and find K with largest score
                K = 2+a.index(max(a))
                knn = KNeighborsClassifier(n_neighbors=K, weights='distance').fit(x_train, y_train)
                y_pred = knn.predict(x_test)
                cvf1['KNeighborsClassifier'], score['KNeighborsClassifier'] = a, f1_score(y_test, y_pred)
            
            # use Logistic Regression
            elif each == 'LogisticRegression':
                a = self.lr_cv(x_train, y_train)
                
                # l begin with 0.5 and find l with largest score
                l = a.index(max(a))*0.5 + 0.5
                lr = LogisticRegression(class_weight='balanced', C=l, max_iter=3000).fit(x_train, y_train)
                y_pred = lr.predict(x_test)
                cvf1['LogisticRegression'], score['LogisticRegression'] = a, f1_score(y_test, y_pred)
            
            # use support vector machine
            elif each == 'SVC':
                a = self.svc_cv(x_train, y_train)
                s = a.index(max(a))
                rf = svm.SVC(class_weight='balanced', kernel=self.range_s[s]).fit(x_train, y_train)
                y_pred = rf.predict(x_test)
                cvf1['SVC'], score['SVC'] = a, f1_score(y_test, y_pred)
                
        return cvf1, score
    
    # get the cross-validation score with all the variable
    def with_wholevar(self, model: list) -> dict:
        return self.get_cvf1_and_score(self.x_data, self.y_data, model)
    
    # get the cross-validation score without a chosen variable
    def without_onevar(self, model: list) -> dict:
        cvf1_without_var = {}
        score_without_var = {}
        
        # get x_data without one column of variable   
        for each in self.origin_col:
            if each != 'id' and each != 'stroke':
                xdata=self.data_without_one_var(self.x_data.copy(), each)
                ydata=self.y_data.copy()
                cvf1_without_var[each], score_without_var[each] = self.get_cvf1_and_score(xdata, ydata, model)
        return cvf1_without_var, score_without_var
    
    # the button for changing the unwanted variable            
    def callback1_var(self, label: str):
            for name, line in lines.items():
                if name == label:
                    line.set_alpha(1)
                else:
                    line.set_alpha(0.2)
            return
        
    # plot the graph of cross-validation score 
    def draw_wholedata(self, cvf1: str):
        global lines, radio1, ax
        range_l = [0,1,2] # = [0.5,1,1.5]
        range_s = [0,1,2] # = ['linear', 'poly', 'rbf']
        fig, ax = plt.subplots()
        plt.ylabel('$cv_f1-score$')
        plt.xlabel('$parameters$')
        
        # adjust position
        plt.subplots_adjust(left=0.42, bottom=0.2)
        
        # create radio box
        radio_ax1 = plt.axes([0.0, 0.5, 0.3, 0.3], facecolor='#FFDDAA')
        
        # create radio buttons
        radio1 = matplotlib.widgets.RadioButtons(radio_ax1, cvf1.keys())
        lines = {}
        
        # choosing different classsifier
        for var, value in cvf1.items():
            if var == 'LogisticRegression':
                r = range_l
            elif var == 'KNeighborsClassifier':
                r = self.range_k
            elif var == 'SVC':
                r = range_s
            if not lines:
                first_variable = var
                lines[var], = ax.plot(r, value, 'o', alpha=1, label=var)
            else:
                lines[var], = ax.plot(r, value, 'o', alpha=1, label=var)
        
        # connect function to radio object and show
        radio1.on_clicked(self.callback1_var)
        ax.legend()
        
        # make sure only first drawing has alpha of 1
        for name, line in lines.items():
            if name == first_variable:
                line.set_alpha(1)
            else:
                line.set_alpha(0.2)
        plt.show()


# In[3]:
# A class only for linear regression
class linear_pred():
    # simple data processing
    def process(self, column: list) -> pd.DataFrame:
        data=pd.read_csv('healthcare-dataset-stroke-data.csv')
        data = data.sample(frac=1).reset_index(drop=True)
        data = data.dropna()
        data=pd.get_dummies(data)

        X = data[column].values.reshape(-1,1)
        y = data['stroke']
        return X,y

    # the function for plotting
    def plotscatterdata (self, x_value: np.ndarray, y_value: np.ndarray, legend: str, x_label: str, y_label: str = 'stroke', alpha: int = 0.5, label_stroke = True ):
        fig, ax = plt.subplots()
        ax.scatter(x_value, y_value, label=legend, alpha = 0.5)
        ax.legend(frameon=False, fontsize='x-large')
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label,fontsize = 15)
        for x, y in zip(x_value, y_value):
            if y == 1:
                plt.text(x, y, '%.0f'%x, ha='center', va='bottom', fontsize=10)
    
    # testing part
    def test(self, column: list,training_fraction: int = 0.9, slice: int = 5):
        Xy = self.process(column)
    
        # further data preprocessing
        # sort the value in each columns
        train_data = list(zip(Xy[0],Xy[1]))
        columns = [column,'stroke']
        train_data = DataFrame(train_data, columns = columns )
        train_data = train_data.sort_values(by=column)
        train_data = train_data.values

        # distribut values randomly into test and train data
        count = 0
        X_train = np.array([])
        y_train = np.array([])
        for i,j in train_data:
            random_number = random.random()
            if random_number < training_fraction:
                X_train = np.append(X_train, i)
                y_train = np.append(y_train, j)
                count += 1
                if count > 5110*training_fraction:
                    break     
        count = 0       
        X_test = np.array([])        
        for i,j in train_data:
            random_number = random.random()
            if random_number < 1-training_fraction:
                X_test = np.append(X_test, i)
                count += 1
                if count > 5110*(1-training_fraction):
                    break

        # split the train and test data into a chosen number of slice
        loc = list(range(0,len(X_train)+1, math.floor(int(len(X_train)/slice))))
        locc = list(range(0,len(X_test)+1, math.floor(int(len(X_test)/slice))))

        # change format of data from list to np.array
        y_linear = np.array([])
        y_logistic = np.array([])
        for i in range (slice): 
            Xtrain = np.array([])
            ytrain = np.array([])
            Xtest = np.array([])
            Xtrain = np.append(Xtrain, X_train[loc[i]:loc[i+1]])
            ytrain = np.append(ytrain, y_train[loc[i]:loc[i+1]])
            Xtest = np.append(Xtest, X_test[locc[i]:locc[i+1]])

            XRAM = DataFrame(Xtrain)
            YRAM = DataFrame(ytrain)

            # linear regrsssion
            regressor = LinearRegression().fit(XRAM.values.reshape(-1,1), YRAM)
            XRAM = DataFrame(Xtest)
            yram = regressor.predict(XRAM.values.reshape(-1,1))
            y_linear = np.append(y_linear, yram)
            
        # presenting data
        self.plotscatterdata(X_test[:len(y_linear)], 
                             y_linear,
                             'linear pred',
                             column,
                             'stroke',
                             True
                       )
        
        # encode the results back into 1 and 0 representing stroke and not stroke
        y_standardized = []
        for y_value in y_linear:
            random_number = random.random()
            if y_value > random_number:
                y_standardized.append(1)
            else:
                y_standardized.append(0)



# In[4]:
# test code
if __name__ == '__main__':
    model=Model('healthcare-dataset-stroke-data.csv', 'stroke')
    
    # pre-processed data
    model.get_data()
    
    # different models used
    models=['LogisticRegression', 'SVC', 'KNeighborsClassifier']
    
    # f1score with whole var
    cvf1,score=model.with_wholevar(models)
    
    # score without one var
    cvf1_without_var,score_without_var=model.without_onevar(models) 
    model.draw_wholedata(cvf1)




