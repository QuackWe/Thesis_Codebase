from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import pickle as pk
import os
from sys import argv

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
dataset_name = argv[1]

model = load_model("dataset/"+dataset_name+"/"+dataset_name+".h5")
pd_len = pd.read_csv("dataset/"+dataset_name+"/len_test"+dataset_name+".csv")
max_len = pd_len['Len'].max()

pickle_test = open("dataset/"+dataset_name+"/"+dataset_name+"_test.pickle","rb")
X_test = pk.load(pickle_test)
image_all = np.asarray(X_test)
image_size = image_all.shape[1]
image_all = np.reshape(image_all, [-1, image_size, image_size, 1])

y_test = pd.read_csv("dataset/"+dataset_name+"/"+dataset_name+"_test_norm.csv")
y_test = y_test[y_test.columns[-1]]
y_test = y_test.astype(int) # ensure integer labels

f = open("dataset/"+dataset_name+"/"+dataset_name+"_results.csv", "w")
f.write("LENGHT;NUMBEROFSAMPLES;ROC_AUC_SCORE"+"\n")

weights = []
list_index_len = []
preds_all = []
preds_all2 = []
y_all = []
i = 0
auc_list = []
f1_score_list = []
index_len = 1
num_classes = model.output_shape[-1]

while i < max_len:
    j = 0
    image_test = []
    target_test = []
    while j < len(pd_len):
        val = pd_len.iloc[j]['Len']
        if val == index_len:
            image_test.append(image_all[j])
            target_test.append(y_test[j])
        j = j + 1
    conv_test = np.asarray(image_test)
    conv_y_test = np.asarray(target_test)
    
    if len(conv_test) == 0:
        # No samples for this length, skip
        i += 1
        index_len += 1
        continue

    pred = model.predict(conv_test)
    pred_classes = np.argmax(pred, axis=1)
    
    preds_all.extend(pred)
    preds_all2.extend(pred_classes)
    y_all.extend(target_test)
    unique_classes = np.unique(conv_y_test)
    if len(target_test) == 1:
        print("skip")
    elif len(unique_classes) <2:
        print(f"Only one class present for length {index_len}, skipping AUC calculation.")
        # Don't call roc_auc_score or append metrics here
    else:
        # Use multi_class='ovr' or 'ovo' for roc_auc_score
        preds_filtered = pred[:, unique_classes]
        if len(unique_classes) == 2:
            # Binary scenario: take the probability of the second class
            auc = roc_auc_score(conv_y_test, preds_filtered[:, 1])  
        else:
            # Multi-class scenario
            auc = roc_auc_score(conv_y_test, preds_filtered, multi_class='ovr')
        f.write(str(index_len)+";"+str(len(target_test))+";"+str(auc)+"\n")
        auc_list.append(auc)
        f1_score_list.append(f1_score(conv_y_test, pred_classes, average='weighted'))
        list_index_len.append(index_len)
        weights.append(len(target_test))
    index_len = index_len + 1
    i = i + 1

auc_list = np.asarray(auc_list)
weights = np.asarray(weights)

auc_weight = np.sum((auc_list * weights))/np.sum(weights)
f1_weight = np.sum((f1_score_list * weights))/np.sum(weights)

print("---METRICS---")
print("WEIGHTED METRICS")
print("ROC_AUC_SCORE weighted: %.2f" % auc_weight)
print("F1_SCORE weighted: %.2f" % f1_weight)

f.write('ROC_AUC_SCORE weighted;'+str(auc_weight)+";"+"\n")
f.write('F1_SCORE weighted;'+str(f1_weight)+";"+"\n")
f.close()

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list_index_len,
    y=auc_list,
    name = dataset_name,
    connectgaps=True
))

fig.update_layout(
    title=dataset_name,
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.show()
