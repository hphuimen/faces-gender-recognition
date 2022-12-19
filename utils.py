# 混淆矩阵
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import auc

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes) - 0.5, -0.5)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


'''y_test=[0,1,1,1,0,0,0,1,1,1,1,0,1,0]
y_pre=[0,1,0,0,0,1,0,1,1,1,0,1,1,0]
class_names = np.array(["female", "male"])  
plot_confusion_matrix(y_test, y_pre, classes=class_names, normalize=False)
plt.show()'''


def accuracy_test(y_test, y_pre):
    acc = accuracy_score(y_test, y_pre)
    return acc


def precision_test(y_test, y_pre):
    precision = precision_score(y_test, y_pre)
    return precision


def recall_test(y_test, y_pre):
    recall = recall_score(y_test, y_pre)
    return recall


def f1_test(y_test, y_pre):
    f1 = f1_score(y_test, y_pre)
    return f1


def plot_roc(y_test, y_pre):
    FPR, TPR, threshhold = metrics.roc_curve(y_test, y_pre, pos_label=1)  # pos_label是定义为正例的标签，与真实值要对应，
    AUC = auc(FPR, TPR)
    plt.plot(FPR, TPR, label='ROC curvem(area = %0.2f)' % AUC, marker='o', color='b', linestyle='-')
    plt.legend(loc='lower right')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
