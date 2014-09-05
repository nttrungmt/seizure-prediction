#!/bin/bash
python seizure_detection.py -b train_data -t Dog_1 2>1 > log/Dog_1 &
python seizure_detection.py -b train_data -t Dog_2 2>1 > log/Dog_2 &
python seizure_detection.py -b train_data -t Dog_3 2>1 > log/Dog_3 &
python seizure_detection.py -b train_data -t Dog_4 2>1 > log/Dog_4 &
python seizure_detection.py -b train_data -t Dog_5 2>1 > log/Dog_5 &
python seizure_detection.py -b train_data -t Patient_1 2>1 > log/Patient_1 &
python seizure_detection.py -b train_data -t Patient_2 2>1 > log/Patient_2 &
