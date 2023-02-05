from metaflow import FlowSpec, step, Parameter, conda, conda_base, IncludeFile
import struct
import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import *

def script_path(filename):
    import os
    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)

def encode_labels(x_train, x_test, index=None):
    label_encoder = sklearn.preprocessing.LabelEncoder()
    df = pd.concat([x_train,x_test],axis=0)
    
    if index == -1:
        print('Encoding y label values')
        not_null_df = df[df.notnull()]
        label_encoder.fit(not_null_df)
        x_train = label_encoder.transform(x_train)
        x_test = label_encoder.transform(x_test)
    
    else:
        print('Encoding X features')
        for i,t in enumerate(df.dtypes):
            if t == 'object':
                s_df = df.iloc[:,i]
                not_null_df = s_df.loc[s_df.notnull()]
                label_encoder.fit(not_null_df)
                try:
                    x_train.iloc[:,i] = x_train.iloc[:,i].astype('float')
                except:
                    x_train.iloc[:,i] = x_train.iloc[:,i].apply(lambda x: label_encoder.transform([x])[0] if x not in [np.nan] else x)
                try:
                    x_test.iloc[:,i] = x_test.iloc[:,i].astype('float')
                except:
                    x_test.iloc[:,i] = x_test.iloc[:,i].apply(lambda x: label_encoder.transform([x])[0] if x not in [np.nan] else x) #np.nan
    return x_train, x_test

def impute_value(x_train, x_test, strategy):
    if strategy == None:
        return x_train.dropna(), x_test.dropna()
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        train_type_dic = dict()
        for i,t in enumerate(x_train.dtypes):
            if t != 'object':
                train_type_dic[i] = t
        test_type_dic = dict()
        for i,t in enumerate(x_test.dtypes):
            if t != 'object':
                test_type_dic[i] = t
        x_train = pd.DataFrame(imp.fit_transform(x_train))
        x_test = pd.DataFrame(imp.transform(x_test))
    return x_train, x_test

def normalize_data(X_train, X_test, scaler = preprocessing.MinMaxScaler()):
    print('Normalized data by scaler')
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def dimension_reduction(x_train, x_test, upper_bound=500, n_components=50,):
    from sklearn.decomposition import PCA
    if x_train.shape[1] >= upper_bound:
        pca = PCA(n_components=n_components, random_state=33)
        pca.fit(x_train)
        x_train= pd.DataFrame(pca.transform(x_train))
        x_test = pd.DataFrame(pca.transform(x_test))
        print("Reducing dimension form %s to %s"%(x_train.shape[1],n_components))
    return x_train, x_test

def read_data(RAWDATA,num_test):
    print("Reading and Cleaning data....")
    columns = ["Page total likes","Type","Category","Post Month","Post Weekday","Post Hour","Paid","Lifetime Post Total Reach","Lifetime Post Total Impressions","Lifetime Engaged Users","Lifetime Post Consumers","Lifetime Post Consumptions","Lifetime Post Impressions by people who have liked your Page","Lifetime Post reach by people who like your Page","Lifetime People who have liked your Page and engaged with your post","comment","like","share","Total Interactions"]
    
    dataframe = dict((column, list()) for column in columns)
    
    # Parse the CSV header.
    lines = RAWDATA.split("\n")
    header = columns
    idx = {column: header.index(column) for column in columns}

    # Populate our dataframe from the lines of the CSV file.
    for line in lines:
        if not line:
            continue
        fields = line.rsplit(";", 19)
        for column in columns:
            dataframe[column].append(fields[idx[column]])
    df = pd.DataFrame(dataframe)
    #df = pd.concat([df.iloc[:,:7],df.iloc[:,-5]],axis=1)
    #df = df.iloc[:,7:]
    X = df.iloc[1:,:-1]
    y = df.iloc[1:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=num_test,random_state=0)
    X_train, X_test = encode_labels(X_train,X_test)
    X_train, X_test = impute_value(X_train, X_test,'mean')
    X_train, X_test = normalize_data(X_train, X_test)
    X_train, X_test = dimension_reduction(X_train, X_test, n_components=15)
    return X_train, X_test, y_train, y_test

class FBRegressionDataExperimentationFlow(FlowSpec):
    fb_data = IncludeFile(
        "fb_data",
        help="The path to a FB metadata file.",
        default=script_path("dataset_facebook.csv"),
    )
    
    num_testing = Parameter('Pct_testing',help='Percentage of Testing Examples',default=0.3,type=float)
    
    @step
    def start(self):
        print("Parameter num_testing has value set to - ",self.num_testing)
        self.train_x,self.test_x,self.train_y,self.test_y = read_data(self.fb_data,self.num_testing)
        # $ Train models in parallel with Sequential, Convolutional and Conv Batch Norm Neural Nets.
        self.next(self.train_svr,self.train_decisiontree,self.train_randomforest,self.train_linearregression)
    
    @step
    def train_svr(self):
        from sklearn.svm import SVR
        from sklearn.gaussian_process.kernels import DotProduct,WhiteKernel,RBF,Matern,RationalQuadratic,ExpSineSquared,ConstantKernel,PairwiseKernel
        print("Training SVR Model.....")
        svr = SVR()
        param_distributions = {
        'kernel' : [DotProduct(),WhiteKernel(),RBF(),Matern(),RationalQuadratic()],
        'C' : scipy.stats.reciprocal(1.0, 10.),
        }
        randcv = RandomizedSearchCV(svr,param_distributions,n_iter=20,cv=3,n_jobs=-1,random_state=0)
        randcv.fit(self.train_x, self.train_y)
        self.train_preds = randcv.predict(self.train_x)
        self.test_preds = randcv.predict(self.test_x)
        self.train_score = sklearn.metrics.r2_score(self.train_y, self.train_preds)
        self.test_score = sklearn.metrics.r2_score(self.test_y, self.test_preds)
        self.next(self.join)
    
    @step
    def train_decisiontree(self):
        from sklearn.tree import DecisionTreeRegressor
        print("Training Decision Tree Model.....")
        tree = DecisionTreeRegressor(random_state=0)
        param_distributions = {
            'max_depth' : scipy.stats.randint(10,100)
        }
        randcv = sklearn.model_selection.RandomizedSearchCV(tree,param_distributions,n_iter=30,cv=3,n_jobs=-1,random_state=0)
        randcv.fit(self.train_x, self.train_y)
        self.train_preds = randcv.predict(self.train_x)
        self.test_preds = randcv.predict(self.test_x)
        self.train_score = sklearn.metrics.r2_score(self.train_y, self.train_preds)
        self.test_score = sklearn.metrics.r2_score(self.test_y, self.test_preds)
        self.next(self.join)
    
    @step    
    def train_randomforest(self):
        from sklearn.ensemble import RandomForestRegressor
        print("Training Random Forest Model.....")
        forest = RandomForestRegressor(random_state=0, warm_start=True)
        param_distributions = {
            'max_depth' : scipy.stats.randint(1,50),
            'n_estimators' : scipy.stats.randint(100,200)
        }
        randcv = sklearn.model_selection.RandomizedSearchCV(forest,param_distributions,n_iter=10,cv=3,n_jobs=-1,random_state=0)
        randcv.fit(self.train_x, self.train_y)
        self.train_preds = randcv.predict(self.train_x)
        self.test_preds = randcv.predict(self.test_x)
        self.train_score = sklearn.metrics.r2_score(self.train_y, self.train_preds)
        self.test_score = sklearn.metrics.r2_score(self.test_y, self.test_preds)
        self.next(self.join)
    
    @step  
    def train_linearregression(self):
        from sklearn.linear_model import LinearRegression
        print('Training LinearRegression ...')
        linear = LinearRegression(n_jobs=-1)
        param_distributions = {
            'normalize' : [True,False]
        }
        randcv = sklearn.model_selection.RandomizedSearchCV(linear,param_distributions,n_iter=2,cv=3,n_jobs=-1,random_state=0)
        randcv.fit(self.train_x, self.train_y)
        self.train_preds = randcv.predict(self.train_x)
        self.test_preds = randcv.predict(self.test_x)
        self.train_score = sklearn.metrics.r2_score(self.train_y, self.train_preds)
        self.test_score = sklearn.metrics.r2_score(self.test_y, self.test_preds)
        self.next(self.join)
    
    @step
    def join(self,inputs):
        """
        Join our parallel branches and merge results,
        """
        print("Joining and merging results...")
        self.test_score = {
            'SVR' : inputs.train_svr.test_score,
            'DS' : inputs.train_decisiontree.test_score,
            'RF' : inputs.train_randomforest.test_score,
            'LR' : inputs.train_linearregression.test_score
        }
        self.trainpredictions = {
            'trainy':inputs.train_svr.train_y,
            'SVR_trainpred':inputs.train_svr.train_preds,
            'DS_trainpred':inputs.train_decisiontree.train_preds,
            'RF_trainpred':inputs.train_randomforest.train_preds,
            'LR_trainpred':inputs.train_linearregression.train_preds
        }
        self.testpredictions = {
            'testy':inputs.train_svr.test_y,
            'SVR_testpred':inputs.train_svr.test_preds,
            'DS_testpred':inputs.train_decisiontree.test_preds,
            'RF_testpred':inputs.train_randomforest.test_preds,
            'LR_testpred':inputs.train_linearregression.test_preds
        }
        self.next(self.end)

    @step
    def end(self):
        """
        This is the end step of the Computation 
        """
        print("Done Computation")

if __name__ == '__main__':
    FBRegressionDataExperimentationFlow()