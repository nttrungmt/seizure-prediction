AUC | Algo | Submission | Comment
--- | --- | --- | ---
0.77097 | Post | 140928-predict.8 | 140928-predict.2 max probabilites | 140928-postprocessing-1
0.76948 | Post | 140928-predict.7 | 140928-predict.2 max probabilites | 140928-postprocessing
0.70747 | RF | 140928-predict.6 | gen-8.5_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 |
0.75955 | Post | 140928-predict.5 | 140928-predict.2 mean probabilites | 140928-postprocessing
0.73419 | Post | 140928-predict.3 | 140928-predict.2 multiply probabilites
0.75501 | Mix | 140928-predict.2 | 140928-predict.1 140926-predict.2 gb=0.4 rf=1 | 140927-mix-submissions
        | GBC | 140928-predict.1 140929-target-combine.validate.1 | gen-8.5_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 n_estimators=1000 pratio=1 in predict | 140927-GBC 140927-GBC-combine
0.75295 | Mix | 140927-predict.2 | 140927-predict.1 140926-predict.2 gb=0.2 rf=1 | 140927-mix-submissions
        | GBC | 140927-predict.1 | gen-8.5_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 n_estimators=100 pratio=5 1 in predict | 140927-GBC 140927-GBC-combine
0.74885 | Mix | 140926-predict.7 | 140926-predict.6 140926-predict.2 gb=0.2 rf=1 | 140926-mix-submissions
        | GBC | 140926-predict.6 | n_estimators=1000 | 140926-GBC 140926-GBC-combine
0.74626 | Mix | 140926-predict.5 | 140926-predict.3 140926-predict.2 gb=0.3 rf=1 | 140926-mix-submissions
0.72241 | Mix | 140926-predict.4 | 140926-predict.3 140926-predict.2 gb=0.7 rf=1 | 140926-mix-submissions
0.63842 | GBC | 140926-predict.3 | n_estimators=400 gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 | 140926-GBC 140926-GBC-combine
0.75037 | RF  | 140926-predict.2 | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9
0.74706 | RF  | 140926-predict.1 | gen-8_medianwindow-bands-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9
0.72947 | RF  | 140923-predict.3 | gen-8_medianwindow2-fft-with-time-freq-corr-2-480-usf-w60
0.73730 | RF  | 140923-predict.2 | gen-8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=0 patient jump
0.67886 | RF  | 140923-predict.1 | gen-8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=0 normal_cutoff = 0.02
0.73781 | RF  | 140922-predict.8 | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=0
0.73779 | RF  | 140922-predict.7 | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=1
0.73235 | RF  | 140922-predict.6 | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=1 jump < 0.7
0.72977 | RF  | 140922-predict.5 | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=0 jump < 0.7
0.72624 | RF  | 140922-predict.4 | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=4
0.73025 | RF  | 140922-predict.3 | medianwindow1-fft-with-time-freq-corr-1-49-r400-usf-w600 PWEIGHT=16
0.66442 | RF  | 140922-predict.2 | medianwindow1-fft-with-time-freq-corr-1-49-r400-usf-w600 PWEIGHT=0
0.72538 | RF  | 140922-predict.1 | medianwindow1-fft-with-time-freq-corr-1-49-r400-usf-w600 PWEIGHT=8
0.73270 | RF  | 140921-predict.5 | gen8_medianwindow-fft-with-time-freq-cov2-1-48-r400-usf-w600
0.73234 | RF  | 140921-predict.4 | gen8_medianwindow-fft-with-time-freq-cov1-1-48-r400-usf-w600
0.71853 | RF  | 140921-predict.3 | gen8_medianwindow-fft-with-time-freq-cov-1-100-r400-usf-w600
0.73397 | RF  | 140921-predict.2 | gen8_medianwindow-fft-with-time-freq-cov-1-48-r400-usf-w600
0.73853 | RF  | 140921-predict.1 | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600
0.70577 | RF  | 140919-predict.2 | gen8_cleancormedianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hammingP2 n_estimators=10000
0.70658 | RF  | 140919-predict.2 | gen8_cleancormedianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hammingP2
0.70117 | RF  | 140919-predict.1 | gen8_cleanmedianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hammingP2
0.68399 | RF  | 140912-rank-calibrate-1 | 
0.68224 | RF  | 140912-rank-calibrate-1 | BUG
0.56332 | RF  | 140912-rank-calibrate | BUG
0.71921 | RF  | 140906-predict-direct.2.3 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False | checkout 7f7b170919be34
0.71528 | RF  | submission1410201279897-rf3000mss1md10Bf_gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.71523 | RF  | 140906-predict-direct.2.2 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False 
0.68699 | RF  | 140907-predict.1 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.71601 | RF  | 140906-predict-direct.2.1 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.70139 | RF  | 140907-predict-direct.2.1 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hamming0 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.70379 | RF  | 140906-predict-direct.3 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hamming2 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.69966 | RF  | 140906-predict-direct.3 | gen16_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.64162 BUG | RF  | 140907-predict-direct.2 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hamming0 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.63918 BUG | RF  | 140907-predict-direct.1 | gen8_medianwindow-fft-with-time-freq-corr-1-96-r400-usf-w600-hamming n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.71775 | RF  | 140906-predict-direct.4 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=6000, min_samples_split=1, max_depth=10 bootstrap=False
0.64970 BUG | RF  | 140906-predict-direct.3 | gen16_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.72043 | RF  | 140906-predict-direct.2 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.70520 | RF  | 140906-predict-direct-1 | gen4_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.69653 | RF  | 140906-predict-direct | gen4_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10
0.68513 | RF  | 140906-predict | 140906-predict-direct + LR calibration
0.70015 | RF  | submission1410030017145-rf3000mss1md10Bf_gen4_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.69119 | RF  | submission1409949301043-rf3000mss1md10Bf_gen2_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.68438 | RF  | submission1409931955461-rf3000mss1md10Bf_gen_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.63495 | RF  | submission1409918444466-rf3000mss1md10Bf_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-0.5-0.9 |
0.64803 | RF  | submission1409867618690-rf3000mss1md10Bf_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 | 
0.64139 | RF  | submission1409858884505-rf3000mss1md10Bf_stdwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.62612 | RF  | submission1409852694072-rf3000mss1md10Bf_window-fft-with-time-freq-corr-1-48-r400-usf-w600 | Buggy run, needs to be re-run
0.61597 | RF  | submission1409838910432-rf1000d6_window-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.61966 | RF  | submission1409829315874-rf3000mss1Bfrs0_window-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.58601 | RF  | submission1409821570759-rf3000mss1Bfrs0_fft-with-time-freq-corr-1-48-r400-usf |
