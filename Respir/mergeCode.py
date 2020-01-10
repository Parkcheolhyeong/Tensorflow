#!/usr/bin/env python3
import csv
import glob
import os

input_path = 'D:\Clouds\git\Respir\DataSet\무호흡'
output_file = '11_merge_csv_files.csv'

is_first_file = True
for input_file in glob.glob(os.path.join(input_path, '*')):
    print(os.path.basename(input_file))
    with open(input_file, 'r', newline='') as csv_in_file:
        with open(output_file, 'a', newline='') as csv_out_file:
            freader = csv.reader(csv_in_file)
            fwriter = csv.writer(csv_out_file)
            if is_first_file:
                for row in freader:
                    fwriter.writerow(row)
                    print(row)
                is_first_file = False
            else:
                header = next(freader)
                for row in freader:
                    fwriter.writerow(row)
                    print(row)


