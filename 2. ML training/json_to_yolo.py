#!/usr/bin/env python
# coding: utf-8



import os
import json


def json_to_yolo(path, folder, file, json_path):
    json_path = path+'/'+folder+'/'+file
    jfile = open(json_path, 'r+', encoding='UTF-8')
    jdata = json.load(jfile)
    jfile.close()
    bound_cnt = int(jdata['BoundingCount'])
    resolution = jdata['RESOLUTION']
    w, h = int(resolution.split('*')[0]), int(resolution.split('*')[1])
    label = set()
    for i in range(bound_cnt):
        try:
            x1 = int(jdata['Bounding'][i]['x1'])
            y1 = int(jdata['Bounding'][i]['y1'])
            x2 = int(jdata['Bounding'][i]['x2']) -                 int(jdata['Bounding'][i]['x1'])
            y2 = int(jdata['Bounding'][i]['y2']) -                 int(jdata['Bounding'][i]['y1'])
            dw = 1. / int(w)
            dh = 1. / int(h)
            x = (float(x1) + float(x1) + float(x2)) / 2.0
            y = (float(y1) + float(y1) + float(y2)) / 2.0
            w = float(x2)
            h = float(y2)

            x = round(x * dw, 6)
            w = round(w * dw, 6)
            y = round(y * dh, 6)
            h = round(h * dh, 6)
        except ZeroDivisionError:
            print(file, 'ZeroDivisionError')
    label.add(file[:2] + ' ' + str(x) + ' ' +
              str(y) + ' ' + str(w) + ' ' + str(h))
    txt_file = path+'/'+folder+'/'+file[:19]+'.txt'
    with open(txt_file, 'w') as f:
        f.write('\n'.join(sorted(label)))

