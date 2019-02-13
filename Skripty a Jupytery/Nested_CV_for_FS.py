#setting path for needed modules
import sys
sys.path.append("../modules")
#custom modules imports
import Classification as CL
import datasets as d
from CrossValForRFandDT import CrossVal
#regular imports
from sklearn.ensemble import RandomForestClassifier,\
AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import os
import progressbar
from itertools import chain
#import warnings

#setting path for saving of validations tabs and for import os
path = os.getcwd()
path = path.replace(path.split('/')[-1],'') # pro windows je třeba "\\" u linuxu stačí "/"
valid_path = path + "/Tabulky a výsledky/Feature Selection/"

bar = progressbar.ProgressBar(maxval=10,#+10,
                              widgets=[progressbar.Bar('#', '[', ']'),
                                       ' ', progressbar.Percentage()])

KOMBINACE = "all"
PARAMS = [[4, 6], [8, 10], [12, 14, 16], range(5,17)]
LR = 0.1
NESTIM = 200

col = ["h-alpha", "1.d sg", "2.d sg"]
col = col + [f"MM {j}" for j in chain([4, 6], [8, 10], [12, 14, 16])]
col = col + [f"EMM {j}" for j in chain([4, 6], [8, 10], [12, 14, 16])]
col = col + [f"MVar {j}" for j in range(5,17)]
#print(col)

if sys.argv[1] == "first":
    DATA = d.load_dataset(name="first_dataset")
elif sys.argv[1] == "second":
    DATA = d.load_dataset(name="second_dataset")
else:
    raise ValueError("Pouze dva datasety: first nebo second!!")

TRAIN, TEST = d.PrepareCrossFold(DATA.data)

FEATURE_IMPORTANCE = []

bar.start()

for cross in range(10):
    train_matrix = CL.make_matrix(TRAIN[cross], KOMBINACE, PARAMS)
    LABELS = d.merge_labels(d.get_layer(TRAIN[cross],2))

    clf = GradientBoostingClassifier(n_estimators=NESTIM, learning_rate=LR)
    #clf = RandomForestClassifier(n_estimators=NESTIM, criterion="entropy", max_depth=3)
    #clf = AdaBoostClassifier(n_estimators=NESTIM, learning_rate=LR)
    clf.fit(train_matrix, LABELS)
    FEATURE_IMPORTANCE.append(np.round(clf.feature_importances_, decimals=4))
    bar.update(cross+1)

TAB = pd.DataFrame(data=np.array(FEATURE_IMPORTANCE), columns=col)
TAB = TAB.transpose()
#TAB.to_csv(valid_path + f"Nested_Feature_Selection_Ada_{sys.argv[1]}.csv")
#TAB.to_csv(valid_path + f"Nested_Feature_Selection_RF_{sys.argv[1]}.csv")
TAB.to_csv(valid_path + f"Nested_Feature_Selection_GB_{sys.argv[1]}.csv")


bar.finish()
print("Hotovo!!!")
