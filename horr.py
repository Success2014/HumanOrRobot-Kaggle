# -*- coding: utf-8 -*-
"""
Created on Thu May 21 02:33:35 2015

@author: Neo
"""

import pandas as pd
import numpy as np

Train = pd.read_csv("E:\PrWS\Kaggle\HorR\MyTrain8.csv")
Test = pd.read_csv("E:\PrWS\Kaggle\HorR\MyTest7.csv")
BadTest = pd.read_csv("E:\PrWS\Kaggle\HorR\BadTest.csv")
#bids = pd.read_csv("E:\PrWS\Kaggle\HorR\MyBids22.csv")


Train = Train[~((Train.outcome==1)&(Train.auc_num==1))] # remove robots with only 1 bid


#----------------------------#
#-------Configurations-------#
#----------------------------#

manualValidation = False
exploreDatabase = False
dataWrangling = False
saveFiles = False

saveResults = True
chooseModel = 1


if manualValidation:
    Train_s_train = Train.loc[32 : 1421]
    Train_s_test = Train.loc[range(0, 32) + range(1422, 1984)]
else:
    Train_s_train = np.array([None] * 10)
    Train_s_test = np.array([None] * 10)


#----------------------------#
#------Database Explore------#
#----------------------------#
if exploreDatabase:
    import MySQLdb
    import sys
    try:
        db = MySQLdb.connect(
                    host = "localhost",
                    user = "root",
                    passwd = "YOUyidi1",
                    db = "facebook")
        
        
    except Exception as e:
        sys.exit("Can't get you into the database you specified!")
        
        
    cursor = db.cursor()
    cursor.execute("select * from train limit 10;")
    results = cursor.fetchall()
    print results



#----------------------------#
#-------Data Wrangling-------#
#----------------------------#
if dataWrangling:
    from dataWrangling import feature_engi
    Train, Test, bids = feature_engi(Train, Test, bids)

#----------------------------#
#--------Saving Files--------#
#----------------------------#
if saveFiles:
    Train.to_csv("MyTrain4.csv")
    Test.to_csv("MyTest3.csv")
 
#----------------------------#
#----Preparing final data----#
#----------------------------#
from prepareFinal import prepareFinalData
train_data, test_data, test_bidder_id, bad_test_bidder_id = prepareFinalData(Train, \
Test, Train_s_train, Train_s_test, manualValidation, BadTest)

#----------------------------#
#-----Build Classifiers------#
#----------------------------#

if chooseModel == 1:

    #---------------------------#
    #----------model 1----------#
    #---------------------------#
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    from tune_parameter import tune_parameter_values
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    
    # make pipeline
    pipeline_rfc = Pipeline(steps = [
                ('scaler', StandardScaler()), # not necessary for RF
                ('classifier', RandomForestClassifier(random_state=66))
    ])
    parameters = {'classifier__n_estimators': list(range(200,250,50)),
                  'classifier__max_features': list(range(5,10,5)),
                  'classifier__max_depth': list(range(5,6,2)),
                  'classifier__min_samples_split': list(range(5,10,5))}
    folds = 10 
    # tune parameter using grid search
    forest = tune_parameter_values(train_data[0::,0], train_data[0::,1::], folds, 
                                       pipeline_rfc, parameters)    
    
    forest = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(max_features = 5,
                                                          n_estimators = 200,                                                          
                                                          max_depth = 5,
                                                          min_samples_split = 5,
                                                          n_jobs = -1,
                                                          random_state=66))
        ])         
    
    # Create the random forest object including all the parameters for the fit
    #forest = RandomForestClassifier(n_estimators = 500, random_state=3)
    # Fit the training data to the Survived labels and create the decision trees
    forest = forest.fit(train_data[0::,1::],train_data[0::,0])
    output = forest.predict_proba(test_data).astype(float)
    output2 = output[0::,1]
    
    # For validation
    if manualValidation:
        output_true = Train_s_test.outcome.values
        print roc_auc_score(output_true, output2)
    
    
elif chooseModel == 2:
    #---------------------------#
    #----------model 2----------#
    #---------------------------#
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    
    params = {'n_estimators': 500, 'max_leaf_nodes': 4, 'max_depth': None, 
              'random_state': 2,'min_samples_split': 5}
    clf = GradientBoostingClassifier(**params)
    clf.fit(train_data[0::,1::],train_data[0::,0])
    output = clf.predict_proba(test_data).astype(float)
    output2 = output[0::,1]
    
    # For validation
    if manualValidation:
        output_true = Train_s_test.outcome.values
        print roc_auc_score(output_true, output2)
    

elif chooseModel == 3:
    #---------------------------#
    #----------model 3----------#
    #---------------------------#
    from sklearn.metrics import roc_auc_score
    from sklearn.naive_bayes import GaussianNB
    
    clf = GaussianNB()
    clf = clf.fit(train_data[0::,1::],train_data[0::,0])
    output = clf.predict_proba(test_data).astype(float)
    output2 = output[0::,1]
    
    # For validation
    if manualValidation:
        output_true = Train_s_test.outcome.values
        roc_auc_score(output_true, output2)

else:
    raise ValueError("Invalid Choice of Model Number")

#---------------------------#
#--------write result-------#
#---------------------------#
if saveResults:
    from writeResults import writeResults
    writeResults(output2, test_bidder_id, bad_test_bidder_id)


