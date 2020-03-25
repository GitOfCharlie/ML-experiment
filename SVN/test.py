import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

true = [0, 0, 0, 0, 1]
pred = [1, 0, 1, 0, 0]
print(precision_score(true, pred, pos_label=0))