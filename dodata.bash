#!/bin/bash
# D5 D1 D2 D4 D3 P1 P2

python seizure_detection.py -b td -t Dog_4 &> log/Dog_4_td &
python seizure_detection.py -b tt -t Dog_4 &> log/Dog_4_tt &

python seizure_detection.py -b td -t Dog_3 &> log/Dog_3_td &
python seizure_detection.py -b tt -t Dog_3 &> log/Dog_3_tt &

python seizure_detection.py -b td -t Patient_1 Dog_5 Dog_1 &> log/Patient_1_Dog_51_td &
python seizure_detection.py -b tt -t Patient_1 Dog_5 Dog_1 &> log/Patient_1_Dog_51_tt &

python seizure_detection.py -b td -t Patient_2 Dog_2 &> log/Patient_2_Dog_2_td &
python seizure_detection.py -b tt -t Patient_2 Dog_2 &> log/Patient_2_Dog_2_tt &
