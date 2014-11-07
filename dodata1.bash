#!/bin/bash
# compute features in parallele on all files (in case you have many cores)

python seizure_detection.py -b td -t Dog_4 &> log/Dog_4_td &
python seizure_detection.py -b tt -t Dog_4 &> log/Dog_4_tt &

python seizure_detection.py -b td -t Dog_3 &> log/Dog_3_td &
python seizure_detection.py -b tt -t Dog_3 &> log/Dog_3_tt &

python seizure_detection.py -b td -t Patient_1 &> log/Patient_1_td &
python seizure_detection.py -b td -t Dog_5 &> log/Dog_5_td &
python seizure_detection.py -b td -t Dog_1 &> log/Dog_1_td &
python seizure_detection.py -b tt -t Patient_1 &> log/Patient_1_tt &
python seizure_detection.py -b tt -t Dog_5 &> log/Dog_5_tt &
python seizure_detection.py -b tt -t Dog_1 &> log/Dog_1_tt &


python seizure_detection.py -b td -t Patient_2 &> log/Patient_2_td &
python seizure_detection.py -b tt -t Patient_2 &> log/Patient_2_tt &
python seizure_detection.py -b td -t Dog_2 &> log/Dog_2_td &
python seizure_detection.py -b tt -t Dog_2 &> log/Dog_2_tt &

