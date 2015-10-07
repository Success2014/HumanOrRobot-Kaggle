# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:08:32 2015

@author: Neo
"""
def writeResults(output2, test_bidder_id, bad_test_bidder_id):
    #import csv
    #prediction_file = open("SubmissionPython_RF3.csv",'wb')
    #prediction_file_object = csv.writer(prediction_file)
    #prediction_file_object.writerow(["bidder_id","prediction"])
    #i = 0
    #for id in test_bidder_id:
    #    prediction_file_object.writerow([id, output2[i]])
    #    i += 1
    #
    #for id in bad_test_bidder_id:
    #    prediction_file_object.writerow([id, 0])
    #prediction_file.close()
    
    
    import csv
    with open("SubmissionPython_RF97.csv","wb") as f:
        writer = csv.writer(f)
        writer.writerow(["bidder_id","prediction"])
        i = 0
        for id in test_bidder_id:
            writer.writerow([id, output2[i]])
            i += 1
        for id in bad_test_bidder_id:
            writer.writerow([id, 0])