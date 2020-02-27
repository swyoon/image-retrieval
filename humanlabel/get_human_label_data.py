"""
get_human_label_data.py
"""
import pandas as pd
import mysql
from mysql import connector

db_info = {'host': 'imlab-ws3.snu.ac.kr',
           'user': 'cbir',
           'passwd': 'cbir@kakao',
           'database': 'CBIR'}

with connector.connect(**db_info) as conn:
    with conn.cursor() as c:
        c.execute("select * from results limit 10")
        a = c.fetchall()

print(a)

