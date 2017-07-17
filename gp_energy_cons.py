# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:27:07 2017

@author: spenc
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, ExpSineSquared, RationalQuadratic
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA as sklearnPCA
from mpl_toolkits.mplot3d import Axes3D
from __future__ import division
from sklearn.externals import joblib

# import the training and test data
# The data is already standardized to a mean of 0 and variance of 1
train_X = np.load('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Project/X_t.npy')
train_Y = np.load('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Project/Y_t.npy')
test_X = np.load('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Project/X_e.npy')
test_Y = np.load('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Project/Y_e.npy')

test_pred = np.load('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Energy_Consumption/GP_test_pred.npy')
test_std = np.load('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Energy_Consumption/GP_test_std.npy')

# Returns the root mean squared error of the parameters
def rmse(y_pred, y):
    return mean_squared_error(y, y_pred)**0.5

# produces a 3D plot when passed a pca dimensionality reduction matrix
def plot_3d(pca, colors = None):
    plt.figure(figsize=(12, 8))
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca[:,0],pca[:,1], pca[:,2], s=15, c=colors)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.show()
    
def plot_y(data, title, xlab, ylab, data_lab = None):
    plt.figure(figsize=(12, 8))
    plt.plot(data, color='blue', label = data_lab)
    plt.suptitle(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.show()
    
def plot_CI(true, pred, sd, ci, hrs):
    choices = {90: 1.645, 95: 1.96, 99:2.58}
    z = choices.get(ci, 'default')
    upper = pred.flatten() + z*sd
    lower = pred.flatten() - z*sd
    plt.figure(figsize=(12, 8))
    plt.fill_between(np.arange(hrs), upper[0:hrs], lower[0:hrs], color='grey', label = str(ci) +'% Confidence Interval')
    plt.scatter(np.arange(hrs), true[0:hrs], c='blue', label = 'True')
    plt.plot(true[0:hrs], c='blue')
    plt.scatter(np.arange(hrs), pred[0:hrs], c='red', label='Predicted')
    plt.legend(fontsize = 'large')
    plt.show()
    in_ci = []
    for idx in xrange(len(true)):
        if  true[idx] > lower[idx] and true[idx] < upper[idx]:
            in_ci.append(1)
        else:
            in_ci.append(0)
    return sum(in_ci) / len(pred)

    
# Plot the entire response variable
plot_y(train_Y, 'All of train_Y', 'Hours', 'Electric load (normalized)')

# Plot the first week to get a closer look at the data
plot_y(train_Y[0:168], 'First week of train_Y', 'Hours', 'Electric load (normalized)', 'Week 1')

# Density plot of the response variable, i.e. the distribution
sns.distplot(train_Y, norm_hist=True, color = 'blue')
sns.plt.suptitle('Density of train_Y')
sns.plt.show()

# Correlation analysis on features - doesn't appear to have any significant correlations
corrs = pd.DataFrame(train_X)
corrs = corrs.corr()
sns.heatmap(corrs, xticklabels=np.arange(12), yticklabels=np.arange(12))


# the following will only work in notebook
#cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
#
#corrs.style.background_gradient(cmap, axis=1)\
#    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
#    .set_precision(2)

# shuffle the data
#rand = np.arange(len(train_X))
#random.shuffle(rand)
#train_Xshuffled = train_X[rand]
#train_Yshuffled = train_Y[rand]


# GP on entire training dataset withouth HMM
kernel = ConstantKernel() + RationalQuadratic() + WhiteKernel()
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = True, n_restarts_optimizer = 5)
gp_model = gp.fit(train_X, train_Y)
joblib.dump(gp_model, 'C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Energy_Consumption/GP_model.pkl') 
gp_model.score(train_X, train_Y)
print rmse(gp_model.predict(train_X), train_Y)

test_pred, test_std = gp_model.predict(test_X, return_std=True)
np.save('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Energy_Consumption/GP_test_pred', test_pred)
np.save('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Energy_Consumption/GP_test_std', test_std)
gp_model.score(test_X, test_Y)
print rmse(test_pred, test_Y)
upper = test_pred.flatten() + 1.96*test_std
lower = test_pred.flatten() - 1.96*test_std

rmse(test_pred, test_Y)
rmse(test_pred, test_Y) / np.mean(test_Y)
rmse(test_pred, test_Y) /(np.max(test_Y) - np.min(test_Y))


#plt.plot(test_Y[0:368], c='red')
#plt.plot(test_pred[0:368], c='blue')   
plot_CI(test_Y, test_pred, 95, 100)

# PCA analysis for visual commparison of hidden states produced
sklearn_pca = sklearnPCA()
train_pca = sklearn_pca.fit_transform(train_X)

plot_3d(train_pca)

# HMM model
hmm = GaussianHMM(n_components=7, min_covar=0.00001, tol = 0.001, n_iter=100).fit(train_X)

# Predict the optimal sequence of hidden states for training data
train_states = hmm.predict(train_X)
cols = ['r', 'b', 'g', 'y', 'k', 'c', 'm', 'darkgreen']
colors = [cols[x] for x in train_states]


zeros = train_X[np.where(train_states == 0)]
zero_Y = train_Y[np.where(train_states == 0)]


# 3D plot to visualize the predicted states 
plot_3d(train_pca, colors)
       

# Implement Gaussian Process for each state, after cross validation on different kernel combinations the
# best kernel for each state was the ConstantKernel() + RationalQuadratic() + WhiteKernel(). The kernel
# hyperparameters are optimized in the call to fit()
#kernel = ConstantKernel() + RationalQuadratic() + WhiteKernel()
#gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = False, n_restarts_optimizer = 5)

models = []
for i in np.arange(7):
    print 'Hidden state ', i
    models.append(gp.fit(train_X[np.where(train_states == i)], train_Yshuffled[np.where(train_states == i)]))


idx = train_X[np.where(train_states == 0)]
gp0 = gp.fit(train_X[np.where(train_states == 0)], train_Yshuffled[np.where(train_states == 0)])
print 'R2 ', gp0.score(train_X[np.where(train_states == 0)], train_Yshuffled[np.where(train_states == 0)])
print rmse(gp0.predict(train_X[np.where(train_states == 0)]), train_Yshuffled[np.where(train_states == 0)])

gp1 = gp.fit(train_X[np.where(train_states == 1)], train_Yshuffled[np.where(train_states == 1)])
print 'R2 ', gp1.score(train_X[np.where(train_states == 1)], train_Yshuffled[np.where(train_states == 1)])
print rmse(gp1.predict(train_X[np.where(train_states == 1)]), train_Yshuffled[np.where(train_states == 1)])

gp2 = gp.fit(train_X[np.where(train_states == 2)], train_Yshuffled[np.where(train_states == 2)])
print 'R2 ', gp2.score(train_X[np.where(train_states == 2)], train_Yshuffled[np.where(train_states == 2)])
print rmse(gp2.predict(train_X[np.where(train_states == 2)]), train_Yshuffled[np.where(train_states == 2)])

gp3 = gp.fit(train_X[np.where(train_states == 3)], train_Yshuffled[np.where(train_states == 3)])
print 'R2 ', gp3.score(train_X[np.where(train_states == 3)], train_Yshuffled[np.where(train_states == 3)])
print rmse(gp3.predict(train_X[np.where(train_states == 3)]), train_Yshuffled[np.where(train_states == 3)])

gp4 = gp.fit(train_X[np.where(train_states == 4)], train_Yshuffled[np.where(train_states == 4)])
print 'R2 ', gp4.score(train_X[np.where(train_states == 4)], train_Yshuffled[np.where(train_states == 4)])
print rmse(gp4.predict(train_X[np.where(train_states == 4)]), train_Yshuffled[np.where(train_states == 4)])

gp5 = gp.fit(train_X[np.where(train_states == 5)], train_Yshuffled[np.where(train_states == 5)])
print 'R2 ', gp5.score(train_X[np.where(train_states == 5)], train_Yshuffled[np.where(train_states == 5)])
print rmse(gp5.predict(train_X[np.where(train_states == 5)]), train_Yshuffled[np.where(train_states == 5)])

gp6 = gp.fit(train_X[np.where(train_states == 6)], train_Yshuffled[np.where(train_states == 6)])
print 'R2 ', gp6.score(train_X[np.where(train_states == 6)], train_Yshuffled[np.where(train_states == 6)])
print rmse(gp6.predict(train_X[np.where(train_states == 6)]), train_Yshuffled[np.where(train_states == 6)])

# test on unseen data
train_states = model.predict(test_X)
y_pred=[]
for i in xrange(len(X)):
    state = train_states[i]
    if state == 0:
        y_pred.append(gp0.predict(X[i], return_std=True))
    elif state == 1:
        y_pred.append(gp1.predict(X[i], return_std=True))
    elif state == 2:
        y_pred.append(gp2.predict(X[i], return_std=True))
    elif state == 3:
        y_pred.append(gp3.predict(X[i], return_std=True))
    elif state == 4:
        y_pred.append(gp4.predict(X[i], return_std=True))
    elif state == 5:
        y_pred.append(gp5.predict(X[i], return_std=True))
    elif state == 6:
        y_pred.append(gp6.predict(X[i], return_std=True))

preds = [x[0][0] for x in y_pred] 
preds = np.array(preds)

gp0 = models[0]
gp0.score(train_X[np.where(train_states == 0)], train_Yshuffled[np.where(train_states == 0)])
y0_pred = gp0.predict(train_X[np.where(train_states == 0)])
plt.plot(Y[0:368], c = 'red')
plt.plot(preds[0:368], c='blue')
plt.show()
for i in np.arange(7):
    print models[i].score(train_X[np.where(train_states == i)], train_Yshuffled[np.where(train_states == i)])

gp0 = gp.fit(train_X[np.where(train_states == 0)], train_Yshuffled[np.where(train_states == 0)])






