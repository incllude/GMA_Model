from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
import numpy as np

train_size = 0.75
random_state = 50


def to_bool(array):
    bias = None
    for i in array:
        if (not np.isclose(0.0, i)) and i < 0.0:
            bias = 0.0
            break
        else:
            bias = 0.5
            break
    return np.array(list(map(lambda x: True if x > bias else False, array)))


def drop_column(df, column):
    return df.drop(column, axis=1), df[column]


def show_results(target, proba):
    if np.array(target).dtype != np.bool_:
        target = to_bool(target)
    predict = to_bool(proba)
    print(f'ROC-AUC score:   {roc_auc_score(np.array(target), np.array(proba))}')
    print(f'Precision score: {precision_score(np.array(target), np.array(predict))}')
    print(f'Recall score:    {recall_score(np.array(target), np.array(predict))}')
    print(f'F1 score:        {f1_score(np.array(target), np.array(predict))}')
    print(f'Accuracy score:  {accuracy_score(np.array(target), np.array(predict))}')
