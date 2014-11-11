  AUC   | Algo |     Submission    | Comment
  --    | ---  |         ---       | ---
0.81204 | Post | 141107-predict.16  | 141107-predict.15 W=0.25 T=0.1 D=-0.5 | 141106-postprocessing
        | Mix  | 141107-predict.15  | 141107-predict.4 141106-predict.1 140926-predict.2 140930-predict.5 gb=0.4 rf=0.9 rf1=0.8 dbn=0.6| 141106-mix-submissions
0.80600 | Post | 141107-predict.14  | 141107-predict.13 W=0.25 T=0.1 D=-0.5 | 141107-fix-GBC 141106-postprocessing
        | Mix  | 141107-predict.13  | 141107-predict.11 141107-predict.2 140926-predict.2 140930-predict.5 gb=0.7 rf=0.9 rf1=0.8 dbn=0.6| 141106-mix-submissions
0.79869 | Post | 141107-predict.12  | 141107-predict.11 W=0.25 T=0.1 D=-0.5 | 141107-fix-GBC 141106-postprocessing
0.81462 | Post | 141107-predict.10  | 141107-predict.4 W=0.25 T=0.1 D=-0.5 | 141106-postprocessing
0.81281 | Post | 141107-predict.9  | 141106-predict.2 W=0.25 T=0.1 D=-0.5 | 141106-postprocessing
        | Mix  | 141107-predict.8  | 141107-predict.4 141107-predict.2 140926-predict.2 140930-predict.5 gb=0.4 rf=0.9 rf1=0.8 dbn=0.6| 141106-mix-submissions
0.81030 | Post | 141107-predict.7  | 141107-predict.6 W=0.25 T=0.1 D=-0.5 | 141106-postprocessing
        | Mix  | 141107-predict.6  | 141107-predict.2 141107-predict.4  rf=0.5 gb=0.5 | 141107-mix-submissions
0.73616 | GB   | 141107-predict.5  | gen-8_maxdiff-60 RAW -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 mask_level=70000 percentile=[0.05, 0.95] nunits=2  | 141107-GBC 141107-GBC-combine
0.76746 | GB   | 141107-predict.4  | gen-8_maxdiff-60 RAW -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 mask_level=70000 percentile=[0.05, 0.95] nunits=2  | 141107-GBC-predict-1
0.76285 | RF   | 141107-predict.3  | gen-8_maxdiff-60 RAW -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 mask_level=10000 percentile=[0.05, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141107-RF-predict
0.76320 | RF   | 141107-predict.2  | gen-8_maxdiff-60 RAW -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 mask_level=7000 percentile=[0.05, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141107-RF-predict
0.76171 | RF   | 141107-predict.1  | gen-8_maxdiff-60 RAW -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141106-RF-predict
0.69981 | RF   | 141106-predict.9  | gen-8_maxdiff-60 141106-data-2-hkl-2 notchwidth=5 -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=3 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141106-RF-predict
0.74750 | RF   | 141106-predict.8  | gen-8_maxdiff-60 141106-data-2-hkl-2 notchwidth=5 -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.5, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141106-RF-predict
0.76012 | RF   | 141106-predict.7  | gen-8_maxdiff-60 141106-data-2-hkl-2 notchwidth=5 -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141106-RF-predict
0.75944 | RF   | 141106-predict.6  | gen-8_maxdiff-60 141106-data-2-hkl-1 notchwidth=5 -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141106-RF-predict
0.75959 | RF   | 141106-predict.5  | gen-8_maxdiff-60 141106-data-2-hkl-1 notchwidth=5 -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141106-RF-predict
0.75653 | RF   | 141106-predict.4  | gen-8_maxdiff-60 141106-data-2-hkl-1 notchwidth=1 -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141106-RF-predict
0.82686 | Post | 141106-predict.3  | 141106-predict.2 W=0.25 T=0.1 D=-0.5 | 141106-postprocessing
        | Mix  | 141106-predict.2  | 140928-predict.1 141106-predict.1 140926-predict.2 140930-predict.5 gb=0.4 rf=0.9 rf1=0.8 dbn=0.6| 141106-mix-submissions
0.76144 | RF   | 141106-predict.1  | gen-8_maxdiff-60 RAW -> gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141106-RF-predict
0.72221 | DBN  | 141104-predict.3  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.1, 0.5, 0.9] nunits=1 300,learn_rates=0.3,learn_rate_decays=0.9,epochs=500 | 141105-DBN-predict-1
0.69981 | DBN  | 141104-predict.3  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 clip=-3,5 percentile=[0.1, 0.5, 0.9] nunits=1 300,learn_rates=0.3,learn_rate_decays=0.9,epochs=500 | 141105-DBN-predict
0.71549 | DBN  | 141104-predict.2  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.1, 0.5, 0.9] nunits=1 300,learn_rates=0.3,learn_rate_decays=0.9,epochs=100 | 141105-DBN-predict
0.70436 | DBN  | 141104-predict.1  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=2 300,learn_rates=0.3,learn_rate_decays=0.9,epochs=100 | 141105-DBN-predict
0.82423 | Post | 141104-predict.13 | 141104-predict.12 W=0.25 T=0.1 D=-0.5 | 141104-postprocessing
0.76768 | Mix  | 141104-predict.12 | 140928-predict.1 141104-predict.8 140926-predict.2 140930-predict.5 gb=0.4 rf=0.8 rf1=0.8 dbn=0.6| 141104-mix-submissions-1
0.81579 | Post | 141104-predict.11 | 141104-predict.10 W=0.25 T=0.1 D=-0.5 | 141104-postprocessing
        | Mix  | 141104-predict.10 | 140928-predict.1 141104-predict.8 141001-predict.1 140930-predict.5 gb=0.4 rf=0.8 rfpca=0.4 dbn=0.6| 141104-mix-submissions
0.74427 | RF   | 141104-predict.9  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=4 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141104-RF-predict-1
0.75805 | RF   | 141104-predict.8  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] nunits=2 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141104-RF-predict-1
0.82054 | Post | 141104-predict.7  | 141104-predict.6 W=0.25 T=0.1 D=-0.5 | 141104-postprocessing
        | Mix  | 141104-predict.6  | 140928-predict.1 141104-predict.5 141001-predict.1 140930-predict.5 gb=0.4 rf=0.8 rfpca=0.4 dbn=0.6| 141104-mix-submissions
0.75344 | RF   | 141104-predict.5  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.05, 0.95] n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141104-RF-predict
0.74223 | RF   | 141104-predict.4  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.2, 0.8] n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141104-RF-predict
0.73296 | RF   | 141104-predict.3  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.9] n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141104-RF-predict
0.75083 | RF   | 141104-predict.2  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.1, 0.9] n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141104-RF-predict
0.74758 | RF   | 141104-predict.1  | gen-8_allbands2-usf-w60-b0.2-b4-b8-b12-b30-b70 percentile=[0.1,0.5,0.9] n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141104-RF-predict
0.82247 | Post | 141103-predict.16 | 141001-predict.4 W=0.25 T=0.1 D=-0.5 | 141103-postprocessing
0.82222 | Post | 141103-predict.15 | 141001-predict.4 W=0.2 T=0.1 D=-0.5 | 141103-postprocessing
0.81156 | Post | 141103-predict.14 | 141103-predict.13 W=0.3 T=0.1 D=-0.5 | 141103-postprocessing
        | Mix  | 141103-predict.13 | 140928-predict.1 141103-predict.12 141001-predict.1 140930-predict.5 gb=0.4 rf=0.8 rfpca=0.4 dbn=0.6| 141103-mix-submissions
0.73706 | RF   | 141103-predict.12 | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False max_depth=10 | 141103-RF-predict
0.81383 | Post | 141103-predict.11 | 141001-predict.4 W=0.3 T=0.1 D=-0.52 | 141103-postprocessing
0.82215 | Post | 141103-predict.10 | 141001-predict.4 W=0.3 T=0.1 D=-0.47 | 141103-postprocessing
0.82107 | Post | 141103-predict.9  | 141001-predict.4 W=0.3 T=0.1 D=-0.4 | 141103-postprocessing
0.80811 | Post | 141103-predict.8  | 141103-predict.7 W=0.3 T=0.1 D=-0.5 | 141103-postprocessing
        | Mix  | 141103-predict.7  | 140928-predict.1 141030-predict.1 141001-predict.1 140930-predict.5 gb=0.4 rf=0.8 rfpca=0.4 dbn=0.6| 141103-mix-submissions
0.82053 | Post | 141103-predict.6  | 141001-predict.4 W=0.4 T=0.1 D=-0.5 | 141103-postprocessing
0.81266 | Post | 141103-predict.5  | 141001-predict.4 W=0.3 T=0.1 D=-0.6 | 141103-postprocessing
0.82075 | Post | 141103-predict.4  | 141001-predict.4 W=0.3 T=0.1 D=-0.3 | 141103-postprocessing
0.82014 | Post | 141103-predict.3  | 141001-predict.4 W=0.3 T=0.1 D=0 | 141103-postprocessing
0.82025 | Post | 141103-predict.2  | 141001-predict.4 W=0.3 T=0.1 D=1 | 141103-postprocessing
0.82246 | Post | 141103-predict.1  | 141001-predict.4 W=0.3 T=0.1 D=-0.5 | 141103-postprocessing
0.76441 | Mix  | 141101-predict.18 | 140928-predict.1 140926-predict.2 141101-predict.14 141001-predict.1 140930-predict.5 gb=0.4 rf=0.7 rf1=0.2 rfpca=0.4 dbn=0.6| 141101-mix-submissions
0.76377 | Mix  | 141101-predict.17 | 140928-predict.1 140926-predict.2 141101-predict.14 141001-predict.1 140930-predict.5 gb=0.4 rf=0.8 rf1=0.3 rfpca=0.4 dbn=0.6| 141101-mix-submissions
0.76041 | Mix  | 141101-predict.16 | 140928-predict.1 140926-predict.2 141101-predict.14 141001-predict.1 140930-predict.5 gb=0.5 rf=0.8 rf1=0.4 rfpca=0.4 dbn=0.4| 141101-mix-submissions
0.76450 | Mix  | 141101-predict.15 | 140928-predict.1 140926-predict.2 141101-predict.14 141001-predict.1 140930-predict.5 gb=0.6 rf=0.8 rf1=0 rfpca=0.4 dbn=0.4| 141101-mix-submissions
0.72018 | RF   | 141101-predict.14 | gen-8.5_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=10 | 141021-RF-predict
0.81390 | Post | 141101-predict.13 | 141001-predict.4 W=0.3 T=0.03  | 141101-postprocessing-1
0.80609 | Post | 141101-predict.12 | 141001-predict.4 W=0.3 T=0.3  | 141101-postprocessing-1
0.78926 | Post | 141101-predict.11 | 141001-predict.4 W=0.3 T=1  | 141101-postprocessing-1
0.81632 | Post | 141101-predict.10 | 141001-predict.4 W=0.3 T=0.1  | 141101-postprocessing-1
0.81195 | Post | 141101-predict.9  | 141001-predict.4 W=0.4 T=0.01  | 141101-postprocessing-1
0.81355 | Post | 141101-predict.8  | 141001-predict.4 W=0.3 (0.7max) T=0.01  | 141101-postprocessing-1
0.78161 | Post | 141101-predict.7  | 141101-predict.5 0.9mean  | 141030-postprocessing
0.79574 | Post | 141101-predict.6  | 141101-predict.5 0.8max  | 141030-postprocessing
0.78239 | Mix  | 141101-predict.5  | 140928-predict.1 141101-predict.3 141001-predict.1 140930-predict.5 gb=0.4 rf=2 rfpca=0.4 dbn=0.6| 141030-mix-submissions
0.74883 | RF   | 141101-predict.4  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False max_depth=10 SMOOTH=0 TRAIN_LOGIT=5 TEST_MIN=.05 TEST_MAX=.9 | 141101-RF-regress-w-test
0.77403 | RF   | 141101-predict.3  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False max_depth=10 SMOOTH=0.3 | 141101-RF-regress-w-test
0.75249 | RF   | 141101-predict.2  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False max_depth=10 SMOOTH=0.5 | 141101-RF-predict-w-test
0.72481 | RF   | 141101-predict.1  | gen-8_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 gen-8_medianwindow1-bands2-usf2-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False max_depth=10 | 141101-RF-predict
0.68140 | RF   | 141031-predict.3  | gen-8_medianwindow1-bands2-usf2-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False max_depth=10 | 141021-RF-predict
0.67035 | RF   | 141031-predict.2  | gen-8_medianwindow1-bands2--w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9-pca min_samples_split=1 bootstrap=False max_depth=10 | 141021-RF-predict
0.69616 | RF   | 141031-predict.1  | gen-8_medianwindow1-bandscorr2--w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9-pca min_samples_split=1 bootstrap=False max_depth=10 | 141021-RF-predict
0.78698 | Post | 141030-predict.9  | 141030-predict.6 1max  | 141030-postprocessing
0.79030 | Post | 141030-predict.8  | 141030-predict.6 0.8max  | 141030-postprocessing
0.78424 | Mix  | 141030-predict.7  | 140928-predict.1 141030-predict.1 141001-predict.1 140930-predict.5 gb=0.4 rf=5 rfpca=0.4 dbn=0.6| 141030-mix-submissions
0.78425 | Mix  | 141030-predict.6  | 140928-predict.1 141030-predict.1 141001-predict.1 140930-predict.5 gb=0.4 rf=3 rfpca=0.4 dbn=0.6| 141030-mix-submissions
0.78378 | Mix  | 141030-predict.5  | 140928-predict.1 141030-predict.1 141001-predict.1 140930-predict.5 gb=0.4 rf=2.4 rfpca=0.4 dbn=0.6| 141030-mix-submissions
0.78278 | Mix  | 141030-predict.4  | 140928-predict.1 141030-predict.1 141001-predict.1 140930-predict.5 gb=0.4 rf=2 rfpca=0.4 dbn=0.6| 141030-mix-submissions
0.78067 | Mix  | 141030-predict.3  | 140928-predict.1 141030-predict.1 141001-predict.1 140930-predict.5 gb=0.4 rf=1.4 rfpca=0.4 dbn=0.6| 141030-mix-submissions
0.77764 | Mix  | 141030-predict.2  | 140928-predict.1 141030-predict.1 141001-predict.1 140930-predict.5 gb=0.4 rf=1 rfpca=0.4 dbn=0.6| 141030-mix-submissions
0.78099 | RF   | 141030-predict.1  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False max_depth=10 | 141030-RF-predict
0.81225 | Post | 141029-predict.10 | 141001-predict.4 0.9max  | 141029-postprocessing-1
0.81119 | Post | 141029-predict.15 | 141001-predict.4 0.6w  | 141029-postprocessing-3
0.81313 | Post | 141029-predict.14 | 141001-predict.4 0.2w  | 141029-postprocessing-3
0.81348 | Post | 141029-predict.13 | 141001-predict.4 0.4w  | 141029-postprocessing-3
0.81224 | Post | 141029-predict.12 | 141001-predict.4 0.4w  | 141029-postprocessing-2
0.80843 | Post | 141029-predict.11 | 141001-predict.4 0.5max  | 141029-postprocessing-1
0.81351 | Post | 141029-predict.10 | 141001-predict.4 0.8max  | 141029-postprocessing-1
0.79907 | Post | 141029-predict.9  | 141001-predict.4 p6 probabilites | 141029-postprocessing
0.79067 | Post | 141029-predict.8  | 141001-predict.4 p5 probabilites | 141029-postprocessing
0.77299 | Post | 141029-predict.7  | 141001-predict.4 p4 probabilites | 141029-postprocessing
0.74477 | RF   | 141029-predict.6  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False max_depth=10 max_features='log2' | 141021-RF-predict
0.76732 | Mix  | 141029-predict.5  | 140928-predict.1 140926-predict.2 141029-predict.4 141001-predict.1 141022-predict.3 140930-predict.5 gb=0.4 rf=0.8 rf1=0.3 rfpca=0.4 rfica=0.35 dbn=0.6| 141022-mix-submissions
0.68846 | RF   | 141029-predict.4  | gen-8_medianwindow1-bandscorr2-nofft-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 maxdepth=10 | 141021-RF-predict
0.76538 | Mix  | 141029-predict.3  | 140928-predict.1 140926-predict.2 141029-predict.1 141001-predict.1 141022-predict.3 140930-predict.5 gb=0.4 rf=0.7 rf1=0.7 rfpca=0.4 rfica=0.35 dbn=0.6| 141022-mix-submissions
0.76551 | Mix  | 141029-predict.2  | 140928-predict.1 140926-predict.2 140929-predict.1 141001-predict.1 141022-predict.3 140930-predict.5 gb=0.4 rf=0.7 rf1=0.7 rfpca=0.4 rfica=0.35 dbn=0.6| 141022-mix-submissions
0.74581 | RF   | 141029-predict.1  | gen-8_medianwindow1-bandscorr2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 maxdepth=10 | 141021-RF-predict
0.67270 | GB   | 141026-predict.6
0.74658 | RF   | 141026-predict.4  | gen-8_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 maxdepth=10 max_features=10 | 141021-RF-predict
0.72120 | RF   | 141026-predict.3  | gen-8_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 gen-8_medianwindow1-bands2--w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False,max_depth=10,max_features=15 | 141021-RF-predict-1
0.71419 | RF   | 141026-predict.2  | gen-8_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 gen-8_medianwindow1-bands2--w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 min_samples_split=1 bootstrap=False,max_depth=10 | 141021-RF-predict-1
0.68272 | RF   | 141026-predict.1  | gen-8_medianwindow1-bands2--w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 | 141021-RF-predict
0.69601 | GB   | 141025-predict.1  | gen-8_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 | 141025-GBC 141025-GBC-combine
0.74831 | RF   | 141024-predict.5  | gen-8_medianwindow1-bands2-usf-w30-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9-s2 | 141021-RF-predict
0.73747 | RF   | 141024-predict.4  | gen-8_medianwindow1-bands2-usf-w30-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 | 141021-RF-predict
0.74865 | RF   | 141024-predict.3  | gen-8_medianwindow1-bands2-usf-w30-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 | 141021-RF-predict
0.63015 | RF   | 141024-predict.2  | gen-8_medianwindow1-bands2-ica-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9-timecorr | 141021-RF-predict
0.64408 | RF   | 141024-predict.1  | gen-8_medianwindow1-bands2-ica-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 | 141021-RF-predict
0.80879 | Post | 141022-predict.6  | 141022-predict.4 max probabilites | 141022-postprocessing
0.76758 | Mix  | 141022-predict.5  | 140928-predict.1 140926-predict.2 141001-predict.1 141022-predict.3 140930-predict.5 gb=0.65 rf=0.8 rfpca=0.4 rfica=0.35 dbn=0.45| 141022-mix-submissions
0.76811 | Mix  | 141022-predict.4  | 140928-predict.1 140926-predict.2 141001-predict.1 141022-predict.3 140930-predict.5 gb=0.4 rf=0.8 rfpca=0.4 rfica=0.35 dbn=0.6| 141022-mix-submissions
0.70404 | RF   | 141022-predict.3  | gen-8_medianwindow1-bands2-141022-ICA-model-1-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 maxdepth=10 |  141016-data-2-hkl-1 141021-RF-predict
0.76625 | Mix  | 141022-predict.2  | 140928-predict.1 140926-predict.2 141001-predict.1 141022-predict.1 140930-predict.5 gb=0.4 rf=0.8 rfpca=0.4 rfica=0.3 dbn=0.6| 141022-mix-submissions
0.65762 | RF   | 141022-predict.1  | gen-8_medianwindow1-bands2-141022-ICA-model-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 maxdepth=10 |  141016-data-2-hkl-1 141021-RF-predict
0.74936 | RF   | 141021-predict.3  | gen-8_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 maxdepth=10 |  141016-data-2-hkl-1 141021-RF-predict
0.70999 | RF   | 141021-predict.2  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 gen-8_medianwindow-bands2-141022-PCA-model-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 max_depth=10 | 141016-data-2-hkl-1 141021-RF-predict-1
0.65619 | RF   | 141021-predict.1  | gen-8_medianwindow-bands2-141022-PCA-model-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 max_depth=5 | 141016-data-2-hkl-1 141021-RF-predict
0.73976 | RF   | 141018-predict.2  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 pratio=1/0 max_evals:100 alpha=0 | 141018-RF-hyperopt-weight
0.71238 | RF   | 141018-predict.1  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 pratio=1/0 max_evals:100 | 141018-RF-hyperopt-weight
0.72598 | RF   | 141017-predict.1  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 pratio=1/0 max_evals:100 | 141017-RF-hyperopt-weight
0.74595 | RF   | 141016-predict.3  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=5 | 141016-data-2-hkl-1 140929-RF-predict
0.74358 | RF   | 141016-predict.2  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=5 | 141016-data-2-hkl 140929-RF-predict
0.74989 | RF   | 141006-predict.4.2 | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=5 | 140929-RF-predict
0.74914 | RF   | 141006-predict.4.1 | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9
0.74957 | RF   | 141006-predict.4  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9
0.74368 | RF   | 141006-predict.3  | gen-8_medianwindow3-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 max_depth=5 | 140929-RF-predict
0.74373 | RF   | 141006-predict.2  | gen-8_medianwindow3-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 max_depth=15 | 140929-RF-predict
0.74710 | RF   | 141006-predict.1  | gen-8_medianwindow3-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 max_depth=10 | 140929-RF-predict
0.80999 | Post | 141005-predict.2  | 141001-predict.4 max probabilites | 141005-postprocessing
0.74741 | RF   | 141005-predict.1  | gen-8_medianwindow2-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-b130-0.1-0.5-0.9 max_depth=10 | 140929-RF-predict
0.56605 | DBN  | 141003-predict.4  | gen-8_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 300,learn_rates=0.3,learn_rate_decays=0.9,dropout=0.1,0.5,epochs=1000 | 140930-DBN-predict
0.61003 | DBN  | 141003-predict.3  | gen-8_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 300,learn_rates=0.3,learn_rate_decays=0.9,dropout=0.5,epochs=1000 | 140930-DBN-predict
0.74985 | RF   | 141003-predict.2  | gen-8_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 max_depth=10 | 140929-RF-predict
0.72810 | RF   | 141003-predict.1  | gen-120_medianwindow1-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 max_depth=10 | 140929-RF-predict
0.76556 | Mix  | 141001-predict.4  | 140928-predict.1 140926-predict.2 141001-predict.1 140930-predict.5 gb=0.4 rf=0.8 rfpca=0.4 dbn=0.6| 141001-mix-submissions
0.77513 | Post | 141001-predict.3  | 141001-predict.2 max probabilites | 140929-postprocessing
0.76473 | Mix  | 141001-predict.2  | 140928-predict.1 140926-predict.2 141001-predict.1 140930-predict.5 gb=0.4 rf=1 rfpca=0.6 dbn=0.6| 141001-mix-submissions
0.71358 | RF   | 141001-predict.1  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9-pca n_estimators=3000, min_samples_split=1, bootstrap=False,max_depth=15 | 140929-RF-predict
0.78043 | Post | 140930-predict.10 | 140930-predict.9 max probabilites | 140929-postprocessing
        | Mix  | 140930-predict.9  | 140928-predict.1 140926-predict.2 140930-predict.5 gb=0.4 rf=1 dbn=0.6| 140930-mix-submissions
0.49115 | DBN  | 140930-predict.8  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 300,learn_rates=0.3,learn_rate_decays=0.9,epochs=500 clip=3| 140930-DBN-predict
0.68890 | DBN  | 140930-predict.7  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 300,learn_rates=0.3,learn_rate_decays=0.9,epochs=1000 | 140930-DBN-predict
        | DBN  | 140930-predict.6  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 300,learn_rates=0.3,learn_rate_decays=0.9,epochs=500 | 140930-DBN-predict
0.71603 | DBN  | 140930-predict.5  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 300,learn_rates=0.3,learn_rate_decays=0.9,epochs=100 | 140930-DBN-predict
0.56280 | DBN  | 140930-predict.4  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 300,learn_rates=0.3,learn_rate_decays=0.9,epochs=10 | 140930-DBN-predict
0.73196 | RF   | 140930-predict.3 gen-8_medianwindow-bands2-usf-w10-hammingP2-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 |  NUNITS=1 maxdepth=20 | 140929-RF-predict
0.73663 | RF   | 140930-predict.2  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 pratio=1/0 | 140930-RF-hyperopt 140929-target-combine
0.73844 | RF   | 140930-predict.1  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-u4-0.1-0.5-0.9 NUNITS=2 maxdepth=5 | 140929-RF-predict
0.74625 | RF   | 140929-predict.4  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-u4-0.1-0.5-0.9 NUNITS=4 maxdepth=5 | 140929-RF-predict
0.75230 | Mix  | 140929-predict.3  | 140928-predict.1 140926-predict.2 gb=0.89 rf=1 | 140929-mix-submissions
0.73932 | RF   | 140929-predict.2  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 pratio=1 | 140929-RF-hyperopt 140929-target-combine
0.68311 | RF   | 140929-predict.1  | gen-8.5_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 pratio=1 | 140929-RF-hyperopt 140929-target-combine
0.77097 | Post | 140928-predict.8  | 140928-predict.2 max probabilites | 140928-postprocessing-1
0.76948 | Post | 140928-predict.7  | 140928-predict.2 max probabilites | 140928-postprocessing
0.70747 | RF   | 140928-predict.6  | gen-8.5_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 |
0.75955 | Post | 140928-predict.5  | 140928-predict.2 mean probabilites | 140928-postprocessing
0.73419 | Post | 140928-predict.3  | 140928-predict.2 multiply probabilites
0.75501 | Mix  | 140928-predict.2  | 140928-predict.1 140926-predict.2 gb=0.4 rf=1 | 140927-mix-submissions
0.73188 | GBC  | 140928-predict.1 140929-target-combine.validate.1 | gen-8.5_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 n_estimators=1000 pratio=1 | 140927-GBC 140927-GBC-combine
0.75295 | Mix  | 140927-predict.2  | 140927-predict.1 140926-predict.2 gb=0.2 rf=1 | 140927-mix-submissions
0.70124 | GBC  | 140927-predict.1  | gen-8.5_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 n_estimators=100 pratio=5/1 in predict | 140927-GBC 140927-GBC-combine
0.74885 | Mix  | 140926-predict.7  | 140926-predict.6 140926-predict.2 gb=0.2 rf=1 | 140926-mix-submissions
0.71844 | GBC  | 140926-predict.6  | n_estimators=1000 | 140926-GBC 140926-GBC-combine
0.74626 | Mix  | 140926-predict.5  | 140926-predict.3 140926-predict.2 gb=0.3 rf=1 | 140926-mix-submissions
0.72241 | Mix  | 140926-predict.4  | 140926-predict.3 140926-predict.2 gb=0.7 rf=1 | 140926-mix-submissions
0.63842 | GBC  | 140926-predict.3  | n_estimators=400 gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9 | 140926-GBC 140926-GBC-combine
0.75037 | RF   | 140926-predict.2  | gen-8_medianwindow-bands2-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9
0.74706 | RF   | 140926-predict.1  | gen-8_medianwindow-bands-usf-w60-b0.2-b4-b8-b12-b30-b70-0.1-0.5-0.9
0.72947 | RF   | 140923-predict.3  | gen-8_medianwindow2-fft-with-time-freq-corr-2-480-usf-w60
0.73730 | RF   | 140923-predict.2  | gen-8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=0 patient jump
0.67886 | RF   | 140923-predict.1  | gen-8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=0 normal_cutoff = 0.02
0.73781 | RF   | 140922-predict.8  | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=0
0.73779 | RF   | 140922-predict.7  | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=1
0.73235 | RF   | 140922-predict.6  | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=1 jump < 0.7
0.72977 | RF   | 140922-predict.5  | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=0 jump < 0.7
0.72624 | RF   | 140922-predict.4  | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600 PWEIGHT=4
0.73025 | RF   | 140922-predict.3  | medianwindow1-fft-with-time-freq-corr-1-49-r400-usf-w600 PWEIGHT=16
0.66442 | RF   | 140922-predict.2  | medianwindow1-fft-with-time-freq-corr-1-49-r400-usf-w600 PWEIGHT=0
0.72538 | RF   | 140922-predict.1  | medianwindow1-fft-with-time-freq-corr-1-49-r400-usf-w600 PWEIGHT=8
0.73270 | RF   | 140921-predict.5  | gen8_medianwindow-fft-with-time-freq-cov2-1-48-r400-usf-w600
0.73234 | RF   | 140921-predict.4  | gen8_medianwindow-fft-with-time-freq-cov1-1-48-r400-usf-w600
0.71853 | RF   | 140921-predict.3  | gen8_medianwindow-fft-with-time-freq-cov-1-100-r400-usf-w600
0.73397 | RF   | 140921-predict.2  | gen8_medianwindow-fft-with-time-freq-cov-1-48-r400-usf-w600
0.73853 | RF   | 140921-predict.1  | gen8_medianwindow1-fft-with-time-freq-corr-1-48-r400-usf-w600
0.70577 | RF   | 140919-predict.2  | gen8_cleancormedianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hammingP2 n_estimators=10000
0.70658 | RF   | 140919-predict.2  | gen8_cleancormedianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hammingP2
0.70117 | RF   | 140919-predict.1  | gen8_cleanmedianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hammingP2
0.68399 | RF   | 140912-rank-calibrate-1 | 
0.68224 | RF   | 140912-rank-calibrate-1 | BUG
0.56332 | RF   | 140912-rank-calibrate | BUG
0.71921 | RF   | 140906-predict-direct.2.3 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False | checkout 7f7b170919be34
0.71528 | RF   | submission1410201279897-rf3000mss1md10Bf_gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.71523 | RF   | 140906-predict-direct.2.2 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False 
0.68699 | RF   | 140907-predict.1 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.71601 | RF   | 140906-predict-direct.2.1 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.70139 | RF   | 140907-predict-direct.2.1 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hamming0 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.70379 | RF   | 140906-predict-direct.3 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hamming2 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.69966 | RF   | 140906-predict-direct.3 | gen16_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.64162 BUG | RF  | 140907-predict-direct.2 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-hamming0 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.63918 BUG | RF  | 140907-predict-direct.1 | gen8_medianwindow-fft-with-time-freq-corr-1-96-r400-usf-w600-hamming n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.71775 | RF   | 140906-predict-direct.4 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=6000, min_samples_split=1, max_depth=10 bootstrap=False
0.64970 BUG | RF  | 140906-predict-direct.3 | gen16_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.72043 | RF   | 140906-predict-direct.2 | gen8_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.70520 | RF   | 140906-predict-direct-1 | gen4_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10 bootstrap=False
0.69653 | RF   | 140906-predict-direct | gen4_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 n_estimators=3000, min_samples_split=1, max_depth=10
0.68513 | RF   | 140906-predict | 140906-predict-direct + LR calibration
0.70015 | RF   | submission1410030017145-rf3000mss1md10Bf_gen4_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.69119 | RF   | submission1409949301043-rf3000mss1md10Bf_gen2_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.68438 | RF   | submission1409931955461-rf3000mss1md10Bf_gen_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.63495 | RF   | submission1409918444466-rf3000mss1md10Bf_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600-0.5-0.9 |
0.64803 | RF   | submission1409867618690-rf3000mss1md10Bf_medianwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 | 
0.64139 | RF   | submission1409858884505-rf3000mss1md10Bf_stdwindow-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.62612 | RF   | submission1409852694072-rf3000mss1md10Bf_window-fft-with-time-freq-corr-1-48-r400-usf-w600 | Buggy run, needs to be re-run
0.61597 | RF   | submission1409838910432-rf1000d6_window-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.61966 | RF   | submission1409829315874-rf3000mss1Bfrs0_window-fft-with-time-freq-corr-1-48-r400-usf-w600 |
0.58601 | RF   | submission1409821570759-rf3000mss1Bfrs0_fft-with-time-freq-corr-1-48-r400-usf |
