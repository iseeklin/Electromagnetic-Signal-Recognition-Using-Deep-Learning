from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(true_data, pre_data, path, cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    cm = confusion_matrix(true_data, pre_data)
    tick_marks = np.arange(len(set(true_data)))
    plt.xticks(tick_marks, fontsize=5.5, rotation=45)
    plt.yticks(tick_marks, fontsize=6)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    print('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True class', fontsize=20,labelpad=12.5)
    plt.xlabel('Predicted class',fontsize=20,labelpad=12.5)
    cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    dig = np.diag(cm1)
    acc = dig.mean()
    acc = format(acc, '.4%')
    print("Mean accuracy:", acc)
    plt.show()
    fig.savefig(path+'Confusion_matrix_'+str(acc)+'.svg')
    fig.savefig(path + 'Confusion_matrix_' + str(acc) + '.jpg')