# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:53:22 2015

@author: Neo
"""

def feature_engi(Train, Test, bids):
    
    #Train['min_bid_time'] = 100000000000000
    #for id in Train.bidder_id:
    #    tmp_df = bids[bids.bidder_id == id]
    #    Train.loc[Train.bidder_id == id,'merchandise_num'] = len(tmp_df.merchandise.unique())
    #    time_diff = (tmp_df['time'] - tmp_df['time'].shift(1))[1:]    
    #    if len(time_diff) > 0:
    #        Train.loc[Train.bidder_id == id, 'min_bid_time'] = min(time_diff)
    #    Train.loc[Train.bidder_id == id, 'total_num'] = tmp_df.shape[0]
    #
    #Test['min_bid_time'] = 100000000000000
    #for id in Test.bidder_id:
    #    tmp_df_test = bids[bids.bidder_id == id]
    #    Test.loc[Test.bidder_id == id,'merchandise_num'] = len(tmp_df_test.merchandise.unique())
    #    time_diff = (tmp_df_test['time'] - tmp_df_test['time'].shift(1))[1:]    
    #    if len(time_diff) > 0:
    #        Test.loc[Test.bidder_id == id, 'min_bid_time'] = min(time_diff) 
    #    Test.loc[Test.bidder_id == id, 'total_num'] = tmp_df_test.shape[0]
    #
    #auction_winner = {}
    #for i in range(bids.shape[0]):
    #    auction_winner[bids.loc[i,'auction']] = bids.loc[i,'bidder_id']
    #
    #Train['success_bid_num'] = 0
    #Test['success_bid_num'] = 0
    #
    #for key, value in auction_winner.items():
    #    if value in (Train.bidder_id.tolist()):
    #        Train.loc[Train.bidder_id == value, 'success_bid_num'] += 1
    #    else:
    #        Test.loc[Test.bidder_id == value, 'success_bid_num'] += 1
    
    
    #all_labeled_robot_id = Train.bidder_id.head(103)
    
    
    #bids.loc[bids.ip=="237.53.38.149"]
    
    #all_train_id = Train.bidder_id.unique()
    #all_test_id = Test.bidder_id.unique()
    #
    #ip_all = []
    #ip_train_num = []
    #ip_test_num = []
    #ip_train_outcome_sum = []
    #ip_train_score = []
    #for aip in bids.ip.unique():
    #    ip_all.append(aip)
    #    ids_of_this_ip = bids.loc[bids.ip == aip].bidder_id.unique()
    #    
    #    ids_of_this_ip_total = len(bids.loc[bids.ip == aip].bidder_id.unique())
    #    
    #    ids_in_train = 0
    #    ip_outocme_sum_tmp = 0
    #    ids_in_test = 0
    #    for each_id in ids_of_this_ip:
    #        if each_id in all_train_id:
    #            ids_in_train += 1
    #            if int(Train.loc[Train.bidder_id==each_id].outcome) == 1:
    #                ip_outocme_sum_tmp += 1
    #        else:
    #            ids_in_test += 1
    #    
    #    assert (ids_of_this_ip_total - ids_in_train == ids_in_test), "ErrorA"
    #    
    #    ip_train_num.append(ids_in_train)
    #    ip_test_num.append(ids_in_test)
    #    ip_train_outcome_sum.append(ip_outocme_sum_tmp)
    #    if ids_in_train > 0:
    #        ip_train_score.append(float(ip_outocme_sum_tmp)/ids_in_train)
    #    else:
    #        ip_train_score.append(22222222)
    #
    #
    #ip_csv = pd.DataFrame({"ip_list":ip_all, "ip_train_num":ip_train_num,
    #                      "ip_test_num":ip_test_num, "ip_train_outcome_sum":ip_train_outcome_sum,
    #                      "ip_train_score":ip_train_score})
    # 
    #ip_csv.to_csv("IP_INFO.csv")
    
    
    
    
    all_bidderid_train = {}
    for id in Train.bidder_id:
        all_bidderid_train[id] = Train.loc[Train.bidder_id==id].outcome.iget(0)
    
    row_outcome = [] # outcome of each row of bids
    for row_index, row in bids.iterrows():    
        row_outcome.append( all_bidderid_train.get(row.bidder_id,0) )
    
    bids['outcome'] = row_outcome
    
    
    in_train = [] # whether each row is in training set
    for row_index, row in bids.iterrows():    
        tmp_scalar = 0
        if all_bidderid_train.get(row.bidder_id, None) is not None:
            tmp_scalar = 1
    
        in_train.append(tmp_scalar)
    
    bids['in_training'] = in_train
    
    
    robot_ip_bank = bids[bids.outcome==1]["ip"].unique()
    robot_url_bank = bids[bids.outcome==1]["url"].unique()
    
    
    train_robot_ip_num = []
    train_robot_url_num = []
    test_robot_ip_num = []
    test_robot_url_num = []
    for id in Train.bidder_id:
        tmp_df = bids[bids.bidder_id == id]
        tmp_robot_ip_count = 0
        tmp_robot_url_count = 0
        for each_ip in tmp_df.ip.unique():
            if each_ip in robot_ip_bank:
                tmp_robot_ip_count += 1
        train_robot_ip_num.append(tmp_robot_ip_count)
        for each_url in tmp_df.url.unique():
            if each_url in robot_url_bank:
                tmp_robot_url_count += 1
        train_robot_url_num.append(tmp_robot_url_count)
    
    for id in Test.bidder_id:
        tmp_df = bids[bids.bidder_id == id]
        tmp_robot_ip_count = 0
        tmp_robot_url_count = 0
        for each_ip in tmp_df.ip.unique():
            if each_ip in robot_ip_bank:
                tmp_robot_ip_count += 1
        test_robot_ip_num.append(tmp_robot_ip_count)
        for each_url in tmp_df.url.unique():
            if each_url in robot_url_bank:
                tmp_robot_url_count += 1
        test_robot_url_num.append(tmp_robot_url_count)
        
    
    Train['robot_ip_num'] = train_robot_ip_num
    Test['robot_ip_num'] = test_robot_ip_num
    Train['robot_url_num'] = train_robot_url_num
    Test['robot_url_num'] = test_robot_url_num
    
    
    Train['country_auc_ratio'] = Train.country_num / Train.auc_num
    Train['ip_auc_ratio'] = Train.ip_num / Train.auc_num
    Train['device_auc_ratio'] = Train.device_num / Train.auc_num
    Train['median_auc_ratio'] = Train.median_b_a_m / Train.auc_num
    Train['max_auc_ratio'] = Train.maxnum_b_a_m / Train.auc_num
    Train['country_ip_ratio'] = Train.country_num / Train.ip_num
    Train['median_ip_ratio'] = Train.median_b_a_m / Train.ip_num
    Train['max_ip_ratio'] = Train.maxnum_b_a_m / Train.ip_num
    Train['success_ip_ratio'] = Train.success_bid_num / Train.ip_num
    Train['country_device_ratio'] = Train.country_num / Train.device_num
    Train['median_device_ratio'] = Train.median_b_a_m / Train.device_num
    Train['max_device_ratio'] = Train.maxnum_b_a_m / Train.device_num
    Train['success_device_ratio'] = Train.success_bid_num / Train.device_num
    
    
    Test['country_auc_ratio'] = Test.country_num / Test.auc_num
    Test['ip_auc_ratio'] = Test.ip_num / Test.auc_num
    Test['device_auc_ratio'] = Test.device_num / Test.auc_num
    Test['median_auc_ratio'] = Test.median_b_a_m / Test.auc_num
    Test['max_auc_ratio'] = Test.maxnum_b_a_m / Test.auc_num
    Test['country_ip_ratio'] = Test.country_num / Test.ip_num
    Test['median_ip_ratio'] = Test.median_b_a_m / Test.ip_num
    Test['max_ip_ratio'] = Test.maxnum_b_a_m / Test.ip_num
    Test['success_ip_ratio'] = Test.success_bid_num / Test.ip_num
    Test['country_device_ratio'] = Test.country_num / Test.device_num
    Test['median_device_ratio'] = Test.median_b_a_m / Test.device_num
    Test['max_device_ratio'] = Test.maxnum_b_a_m / Test.device_num
    Test['success_device_ratio'] = Test.success_bid_num / Test.device_num
    
    return Train, Test, bids

