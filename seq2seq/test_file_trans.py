# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 1:32 AM
# @Author  : SiJin Wu
# @FileName: test_file_trans.py
import csv

with open("test_preds.csv", 'r') as f, open("test_preds_upload.csv", 'w') as fout:
    csv_reader = csv.reader(f, delimiter='\t')
    csv_writer = csv.writer(fout, delimiter=',', quoting=csv.QUOTE_NONE, quotechar='')
    csv_writer.writerow(["decomposition"])
    for p in csv_reader:
        p[0] = p[0].replace(",", "").replace("returnreturn", "return").replace("????????return", "return").replace("Newsreturn", "return").replace("'return", "return")
        csv_writer.writerow(p)
