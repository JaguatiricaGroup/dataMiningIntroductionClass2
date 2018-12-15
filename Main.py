# from Cython.Includes.libcpp.iterator import insert_iterator
import Analises as an
from LeInstancia import LeInstancia
from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import logistic
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.cluster import KMeans
# import Grafico3D as graf
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn import svm, datasets
# from sklearn.metrics import confusion_matrix

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score

def regressaoLogistica(X_train, X_test, y_train):
    gnb = linear_model.LogisticRegression()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    return y_pred, gnb

def regressaoLinear(X, Y):
    modelo = LinearRegression()
    y_pred = modelo.fit(X, Y).predict(X)
    return y_pred, modelo


def suportVectorMachine(X_train, X_test, y_train):
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    # print(classifier.coef0)

    return y_pred, classifier


def naiveBayes(X_train, X_test, y_train):
    gnb = GaussianNB()
    # y_pred = gnb.fit(data[:8562], target[:8562]).predict(data[8562:])
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    with open('infosNaiveBayes.txt', 'w') as arquivo:
        arquivo.write(
            'quantidade' + '\n' + 'pele: ' + str(gnb.class_count_[0]) + '\n' + 'Não pele: ' + str(gnb.class_count_[1]))
        arquivo.write('quantidade' + '\n' + 'pele: ' + str(gnb.priors))

    return y_pred, gnb


def matrizConfusa(y_test, y_pred, nome):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    an.plot_confusion_matrix(cnf_matrix, class_names, nome, normalize=True,
                             title='Normalized confusion matrix')
    plt.savefig('MatrizConfusa_' + nome + '.pdf')
    plt.close()

def plotaRegressao(data, target):
    predito, modelo = regressaoLinear(data, target)
    # print(regressaoLinear(data, target))
    print(modelo.coef_)
    b = modelo.intercept_
    a = modelo.coef_[0]
    y = lambda x : b + a*x
    dominio = np.arange(0,255)
    plt.plot(y(dominio), color='green')
    plt.scatter(data, target)
    plt.show()

def executaAnalises(X_train, X_test, y_train, y_test):
    nome = 'naiveBayes'
    previsao, classifier = naiveBayes(X_train=X_train, X_test=X_test, y_train=y_train)
    matrizConfusa(y_test, previsao, nome)
    an.AUC(previsao, y_test, nome)
    # an.Plota3D(X_test, y_test, classifier, nome)

    nome = 'SVM'
    previsao, classifier = suportVectorMachine(X_train=X_train, X_test=X_test, y_train=y_train)
    matrizConfusa(y_test, previsao, nome)
    an.AUC(previsao, y_test, nome)
    an.Plota3D(X_test, y_test, classifier, nome)

    nome = 'RegressaoLogistica'
    previsao, classifier = regressaoLogistica(X_train=X_train, X_test=X_test, y_train=y_train)
    matrizConfusa(y_test, previsao, nome)
    an.AUC(previsao, y_test, nome)
    an.Plota3D(X_test, y_test, classifier, nome)

def plotElbow(X):
    elbows = list()
    lim = 10
    for n_cluster in range(2, lim):
        kmeans = KMeans(n_clusters=n_cluster).fit(X)
        label = kmeans.labels_
        elbows.append(silhouette_score(X, label, metric='euclidean'))
    x = np.arange(2,lim)
    y = elbows
    plt.plot(x, y)
    plt.savefig('GraficoElbow.pdf')
    plt.close()
    return max(elbows)


def plotCluster(X):
    numeroClusters = 8
    kmeans = KMeans(numeroClusters)
    kmeans = kmeans.fit(X)
    classes = kmeans.predict(X)
    plt.scatter(X['b'], X['g'], c=classes)
    plt.savefig('GraficoClusterizado_numCluster_'+str(numeroClusters)+'.pdf')
    plt.close()
# prepara os trem

instancia = LeInstancia('database-pele-06.dat')
# instancia.descreve()

instancia.removeOutlier('b')
instancia.removeOutlier('g')

class_names = ['pele', 'npele']
data = instancia.baseDados[['b']].dropna().get_values()
target = instancia.baseDados['g'].dropna().get_values()
# X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)
# print('tamanho do teste: ', len(X_test), 'tamanho do treino', len(X_train))
# plotaRegressao(data, target)
# plt.plot()
X = instancia.baseDados[['b','g']]
# xCluster = np.array(data, target)
# print(xCluster)
# plotElbow(X)
plotCluster(X)
# Silhouette


# blue e green





# y_pred, classifier = suportVectorMachine(X_train, X_test, y_train)

# # analises
# instancia.boxplot()
# # instancia.descreve()

# # executaMatrizesConf(X_train, X_test, y_train, y_test)
# print(len(X_test))
# executaAnalises(X_train, X_test, y_train, y_test )
# an.Plota3D(X_test, y_test, classifier, 'Svm')





# target = instancia.baseDados['pele'].dropna().get_values()
# an.plot_confusion_matrix(previsao, target[8562:])
#
#
# h = .02  # step size in the mesh
#
#
#
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# b_min, b_max = instancia.baseDados['b'].min() - 0.5, instancia.baseDados['b'].max() - 0.5
# g_min, g_max = instancia.baseDados['g'].min() - 0.5, instancia.baseDados['g'].max() - 0.5
#
# r_min, r_max = instancia.baseDados['r'].min() - 0.5, instancia.baseDados['r'].max() - 0.5
#
# bb, gg, rr = np.meshgrid(np.arange(b_min, b_max, h), np.arange(g_min, g_max, h),np.arange(r_min, r_max, h))
# Z = regressaoLogistica(instancia)#logreg.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(bb.shape)
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(bb, gg,rr, Z, cmap=plt.cm.Paired)
#
# # Plot also the training points
# plt.scatter(instancia.baseDados['b'], instancia.baseDados['g'], instancia.baseDados['r'],c=instancia.baseDados['pele'], edgecolors='k', cmap=plt.cm.Paired)
# #plt.xlabel('Sepal length')
# #plt.ylabel('Sepal width')
#
# plt.xlim(bb.min(), bb.max())
# plt.ylim(gg.min(), gg.max())
# #plt.zlim(rr.min(),rr.max())
# plt.xticks(())
# plt.yticks(())
# #plt.zticks(())
#
# plt.show()
#
# instancia.normalizaTodos()
# # gnb = GaussianNB()
# # data = instancia.baseDados[['b', 'g', 'r']].dropna().get_values()
# # target = instancia.baseDados['pele'].dropna().get_values()
# # print('ou to vazio', target)
# # target = instancia.baseDados['pele'].values()
# instancia.matrizCorrelacao()
# instancia.descreve()
# regressaoLogistica(instancia)
# print("Number of mislabeled points out of a total %d points : %d"
#       % (data.shape[0], (target != y_pred).sum()))
