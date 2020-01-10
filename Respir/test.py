#!/usr/bin/env python3
import csv
import glob
import os
import pandas as pd
input_path = 'D:\Clouds\git\Respir\DataSet\무호흡'
output_file = '11_merge_csv_files.csv'


df_temp = pd.read_csv('D:\Clouds\git\Respir\DataSet\무호흡\log_0701_1518.csv')
print(df_temp)
