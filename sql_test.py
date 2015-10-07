# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:13:06 2015

@author: Neo
"""

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