#!/bin/bash
# D5 D1 D2 D4 D3 P1 P2
echo $1
rm log/*
python seizure_detection.py -b $1 -t Dog_1 &> log/Dog_1 &
python seizure_detection.py -b $1 -t Dog_2 &> log/Dog_2 &
python seizure_detection.py -b $1 -t Dog_3 &> log/Dog_3 &
python seizure_detection.py -b $1 -t Dog_4 &> log/Dog_4 &
python seizure_detection.py -b $1 -t Dog_5 &> log/Dog_5 &
python seizure_detection.py -b $1 -t Patient_1 &> log/Patient_1 &
python seizure_detection.py -b $1 -t Patient_2 &> log/Patient_2 &
