import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings(action='ignore')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import os

gedi_vars = pd.read_csv('gediVars.csv')

gedi_vars_sel = gedi_vars[gedi_vars['gedi_both'] == 1]; gedi_vars_sel = gedi_vars_sel[gedi_vars_sel.b_gedi_num>2]; gedi_vars_sel = gedi_vars_sel[gedi_vars_sel.a_gedi_num>2]

drop_col = ['id','left','top','right','bottom','a_elev_m','a_dem_m','a_celev_m','a_ss_m', 'a_mfrac_m','a_rv_m', 'a_rg_m','a_rxrtrn_m','b_mfrac_m','b_rv_m','b_elev_m','b_dem_m','b_celev_m','b_ss_m', 'b_rg_m','b_rxrtrn_m','gedi_both','a_gedi_num','b_gedi_num', 'a_crh000_m', 'b_crh000_m', 'iscontrol']

gedi_vars_sel_vals = gedi_vars_sel.drop(columns = drop_col)

a_val_cols = ['a_crh100_m', 'a_crh095_m', 'a_crh090_m', 'a_crh085_m', 'a_crh080_m', 'a_crh075_m', 'a_crh070_m', 'a_crh065_m', 'a_crh060_m', 'a_crh055_m', 'a_crh050_m', 'a_crh045_m', 'a_crh040_m', 'a_crh035_m', 'a_crh030_m', 'a_crh025_m', 'a_crh020_m', 'a_crh015_m', 'a_crh010_m', 'a_crh005_m', 'a_cv_m', 'a_fhd_m', 'a_pai_m', 'a_gap_m', 'a_pavd01_m', 'a_pavd02_m', 'a_pavd03_m', 'a_pavd04_m', 'a_pavd05_m', 'a_pavd06_m', 'a_pavd07_m']

gedi_vars_sel_vals_delta = gedi_vars_sel_vals[['isburnt']]

for col in a_val_cols:
    pre = col.replace('a_', 'b_')
    post = col
    delta = col.replace('a_', 'd_')
    gedi_vars_sel_vals_delta[delta] = gedi_vars_sel_vals[pre] - gedi_vars_sel_vals[post]

x = gedi_vars_sel_vals_delta[gedi_vars_sel_vals_delta.columns.difference(['isburnt'])]
y = gedi_vars_sel_vals_delta['isburnt']

differ_cols = ['isburnt', "d_gap_m", 'd_pavd03_m', 'd_crh040_m', 'd_crh060_m', 'd_crh050_m', 'd_crh030_m', 'd_pai_m', 'd_crh080_m', 'd_crh070_m', 'd_crh020_m', 'd_crh055_m', 'd_crh035_m', 'd_crh090_m', 'd_crh075_m', 'd_crh015_m', 'd_crh095_m', 'd_crh045_m', 'd_crh010_m', 'd_crh085_m', 'd_pavd05_m', 'd_crh025_m', 'd_cv_m', 'd_crh065_m', 'd_pavd07_m']
considered_features = gedi_vars_sel_vals_delta.columns.difference(differ_cols)

x_train, x_test, y_train, y_test = train_test_split(x[considered_features], y, test_size = 0.3, stratify=y, random_state = 1234)

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train.value_counts(normalize=True)

svm_clf = SVC(kernel='rbf', gamma=0.075, C=4000)
svm_clf.fit(x_train, y_train)

y_pred = svm_clf.predict(x_test)

cf_matrix = confusion_matrix(y_test, y_pred)

tn = cf_matrix[0][0]; fp = cf_matrix[0][1]; fn = cf_matrix[1][0]; tp = cf_matrix[1][1]

accuracy = (tp+tn)/(tp+tn+fp+fn);precision = tp/(tp+fp); recall = tp/(tp+fn); f1 = (2*precision*recall)/(precision+recall)

sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

print(accuracy, precision, recall, f1)

x_scaled = scaler.fit_transform(x[considered_features])

y_pred = svm_clf.predict(x_scaled)

cf_matrix = confusion_matrix(y, y_pred)

tn = cf_matrix[0][0]; fp = cf_matrix[0][1]; fn = cf_matrix[1][0]; tp = cf_matrix[1][1]

accuracy = (tp+tn)/(tp+tn+fp+fn);precision = tp/(tp+fp); recall = tp/(tp+fn); f1 = (2*precision*recall)/(precision+recall)

sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

print(accuracy, precision, recall, f1)