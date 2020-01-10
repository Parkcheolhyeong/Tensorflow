#!/usr/bin/env python3
import csv
import glob
import os
import pandas as pd
#input_path = 'D:\Clouds\git\Respir\DataSet\비정상호흡'
#input_path = 'D:\Clouds\git\Respir\DataSet\정상호흡\일반_정상'
#input_path =  'D:\Clouds\git\Respir\DataSet\Test\정상'
input_path = 'D:\Clouds\git\Respir\DataSet\Test\무호흡'

output_file = '11_merge_csv_files.csv'
#time_pd2.to_csv("filename.csv", mode='a', header=False)
#헤더 안불러옴

df = pd.DataFrame({'0': []})
global count
count = 0
is_first_file = True
for input_file in glob.glob(os.path.join(input_path, '*.csv')):
    #print(os.path.basename(input_file))
    a = str(os.path.basename(input_file))
    #print(a)

    path = input_path+'\\'+a
    b = a.split('.')
    sheetName = str(b[0])

    print(path)
    print(sheetName)

    df_temp = pd.read_csv(path, names = ["A", "B", "C"])
    print(df_temp)
    df[str(count)] = df_temp["A"]
    count =count + 1
    if count == 60:
        break

#df.to_csv("Apnea.csv", index=False, header=False, mode='w')
#df.to_csv("Respir.csv", index=False, header=False, mode='w')
#df.to_csv("Respir_test.csv", index=False, header=False, mode='w')
df.to_csv("Apnea_test.csv", index=False, header=False, mode='w')

#print(df)