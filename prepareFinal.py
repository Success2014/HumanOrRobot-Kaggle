# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 14:59:23 2015

@author: Neo
"""


def prepareFinalData(Train, Test, Train_s_train, Train_s_test, manualValidation, BadTest):
    #Train = Train.drop(['X','bidder_id','payment_account','address','merchandise_num'],axis = 1)
    #Bad_Test_data = Test.loc[Test.auc_num == 0,]
    #Test = Test.loc[Test.auc_num != 0, ]
    #test_bidder_id = Test.bidder_id
    #Test = Test.drop(['X','bidder_id','payment_account','address','merchandise_num'], axis = 1)
    
    
    #Train_final = Train_s_train[['outcome','country_num','ip_num','device_num',
    #'median_b_a_m','maxnum_b_a_m','auc_num','min_bid_time','success_bid_num',
    #'total_bids_num','total_auc_ratio', 'max_median_prdt','success_auc_ratio', 
    #'success2_auc_ratio', 'median_country_ratio','ip_device_ratio']]
    #Test_final = Train_s_test[['country_num','ip_num','device_num','median_b_a_m',
    #'maxnum_b_a_m','auc_num','min_bid_time','success_bid_num','total_bids_num',
    #'total_auc_ratio','max_median_prdt','success_auc_ratio', 'success2_auc_ratio', 
    #'median_country_ratio','ip_device_ratio']]
    
    bad_test_bidder_id = BadTest.bidder_id
    
    if manualValidation:
        Train_final = Train_s_train.drop(['bidder_id', 'payment_account', 'address','robot_ip_num','robot_url_num'], axis = 1)
        Test_final = Train_s_test.drop(['outcome','bidder_id', 'payment_account', 'address','robot_ip_num','robot_url_num'], axis = 1)
    else:
        Train_final = Train.drop(['bidder_id', 'payment_account', 'address','robot_ip_num','robot_url_num'], axis = 1)
        Test_final = Test.drop(['bidder_id', 'payment_account', 'address','robot_ip_num','robot_url_num'], axis = 1)
    
    
    test_bidder_id = Test.bidder_id
    
    train_data = Train_final.values
    test_data = Test_final.values
    
    return train_data, test_data, test_bidder_id, bad_test_bidder_id