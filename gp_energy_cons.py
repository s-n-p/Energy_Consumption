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
plt.figure(figsize=(12, 8))
g = sns.distplot(train_Y, norm_hist=True, color = 'blue')
g.figure.suptitle('Density of train_Y')


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

# Instantiate and fit the model
gp_model = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = True, n_restarts_optimizer = 5).fit(train_X, train_Y)

# Save the model to a file for future reference
#joblib.dump(gp_model, 'C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Energy_Consumption/GP_model.pkl') 
# to load model run
#gp_model = joblib.load('GP_model.pkl') 

# Get predictions and standard deviations of the model on train_X
train_pred, train_std = gp_model.predict(train_X, return_std=True)

#print the r-squared ad rmse of the predictions on training data
print 'R-squared: ', gp_model.score(train_X, train_Y)
print 'RMSE: ', rmse(train_pred, train_Y)

# Plot the true values of train_Y vs model predictions on train_Y
plot_y([train_Y, train_pred],  'True values vs. model predictions on test data', 'Hours', 'Electric load (normalized)', ['Truth','Prediction'])
in_ci = plot_CI('Confidence Interval on training data', train_Y, train_pred, train_std, 95, 250, 450)
print 'The percent of true values that fall within the confidence interval is ', 100*in_ci,'%'

# Density plot of the test response variable and the predicted values
plt.figure(figsize=(12, 8))
sns.distplot(train_Y, norm_hist=True, color = 'blue', label = 'True')
sns.distplot(train_pred, norm_hist=True, color = 'red', label = 'Predicted')
plt.suptitle('Density of true vs. predicted values on training data')
plt.legend()
plt.show()

# get the predictions and standard deviation of test data
test_pred, test_std = gp_model.predict(test_X, return_std=True)

# Save the predictions and std deviation
np.save('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Energy_Consumption/GP_test_pred', test_pred)
np.save('C:/Users/spenc/Documents/COURSES/CS6961_StructuredPrediction/Energy_Consumption/GP_test_std', test_std)

#print the r-squared ad rmse
print 'R-squared: ', gp_model.score(test_X, test_Y)
print 'RMSE: ', rmse(test_pred, test_Y)

# View plots of the true test_Y values vs the predictions 
plot_y([test_Y, test_pred],  'True values vs. model predictions on unseen (test_X) data', 'Hours', 'Electric load (normalized)', ['Truth','Prediction'])
in_ci = plot_CI('Confidence Interval on unseen (test_X) data', test_Y, test_pred, test_std, 95, 200, 325)
print 'The percent of true values that fall within the confidence interval is ', 100*in_ci,'%'

# Density plot of the test response variable and the predicted values
plt.figure(figsize=(12, 8))
sns.distplot(test_Y, norm_hist=True, color = 'blue', label = 'True')
sns.distplot(test_pred, norm_hist=True, color = 'red', label = 'Predicted')
plt.suptitle('Density of true vs. predicted values on unseen data')
plt.legend()
plt.show()

######
######  Hidden Markov with GP regression model  ##################################################
######

# PCA analysis for visual commparison of hidden states produced
sklearn_pca = sklearnPCA()
train_pca = sklearn_pca.fit_transform(train_X)

# plot the first 3 components 
plot_3d(train_pca)

# HMM model
hmm = GaussianHMM(n_components=7, min_covar=0.00001, tol = 0.001, n_iter=100).fit(train_X)

# Predict the optimal sequence of hidden states for training data
train_hidden_states = hmm.predict(train_X)
cols = ['r', 'b', 'g', 'y', 'k', 'c', 'm', 'darkgreen']
colors = [cols[x] for x in train_states]

# 3D plot of the training data colored with the predicted hidden state for each data point.
print '3D plot of the first 3 PCA components colored by hidden state that produced the data point:'
plot_3d(train_pca, colors)


print 'Plot of train_Y, the response variable, colored by the hidden state'
# plot the train_X data with the hidden states color coded
plt.figure(figsize=(12, 8))
plt.plot(train_Y[0:200])
plt.scatter(np.arange(200), train_Y[0:200], c=colors)
plt.show()      

# Implement Gaussian Process for each state, after cross validation on different kernel combinations the
# best kernel for each state was the ConstantKernel() + RationalQuadratic() + WhiteKernel(). The kernel
# hyperparameters are optimized in the call to fit()
#kernel = ConstantKernel() + RationalQuadratic() + WhiteKernel()
#gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = False, n_restarts_optimizer = 5)

def r2_rmse(model, state):
    print ' GP for Hidden state ', state
    print 'R-squared: ', model.score(train_X[np.where(train_hidden_states == state-1)], train_Y[np.where(train_hidden_states == state-1)])
    print 'RMSE: ', rmse(model.predict(train_X[np.where(train_hidden_states == state-1)]), train_Y[np.where(train_hidden_states == state-1)])

models = []
for i in np.arange(7):
    print 'Hidden state ', i
    models.append(gp.fit(train_X[np.where(train_states == i)], train_Yshuffled[np.where(train_states == i)]))


gp_hmm0 = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = True, n_restarts_optimizer = 5).fit(
    train_X[np.where(train_hidden_states == 0)], train_Y[np.where(train_hidden_states == 0)])
gp_hmm1 = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = True, n_restarts_optimizer = 5).fit(
    train_X[np.where(train_hidden_states == 1)], train_Y[np.where(train_hidden_states == 1)])
gp_hmm2 = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = True, n_restarts_optimizer = 5).fit(
    train_X[np.where(train_hidden_states == 2)], train_Y[np.where(train_hidden_states == 2)])
gp_hmm3 = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = True, n_restarts_optimizer = 5).fit(
    train_X[np.where(train_hidden_states == 3)], train_Y[np.where(train_hidden_states == 3)])
gp_hmm4 = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = True, n_restarts_optimizer = 5).fit(
    train_X[np.where(train_hidden_states == 4)], train_Y[np.where(train_hidden_states == 4)])
gp_hmm5 = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = True, n_restarts_optimizer = 5).fit(
    train_X[np.where(train_hidden_states == 5)], train_Y[np.where(train_hidden_states == 5)])
gp_hmm6 = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y = True, n_restarts_optimizer = 5).fit(
    train_X[np.where(train_hidden_states == 6)], train_Y[np.where(train_hidden_states == 6)])

# dictionary of hidden state and corresponding GP model
models = {1:gp_hmm0, 2:gp_hmm1, 3:gp_hmm2, 4:gp_hmm3, 5:gp_hmm4, 6:gp_hmm5, 7:gp_hmm6}

# Look at R-squared, rmse, and density plots of true vs predicted on train_X
fig = plt.figure(figsize=(18, 15))
for i in np.arange(1, 8):
    r2_rmse(models[i], i)
    print '-'*20
    fig.add_subplot(4,2,i, title='Hidden state '+str(i))
    sns.distplot(train_Y[np.where(train_hidden_states == i-1)], norm_hist=True, color='blue',label = 'true')
    sns.distplot(models[i].predict(train_X[np.where(train_hidden_states == i-1)]), norm_hist=True,  color='red',label = 'pred')
    plt.suptitle('Density of true vs. predicted values for each state on training data')
    plt.legend(loc=1)
    
plt.show()

# predict the most likely state sequence for the unseen test data
test_hidden_states = hmm.predict(test_X)

# For each observation get the predicted hidden state and make a energy consumption prediction using the GP model for that state
test_pred_hmm_gp = []
test_std_hmm_gp = []
for idx in range(len(test_X)):
    state = test_hidden_states[idx]+1
    pred, std = models[state].predict(test_X[idx], return_std=True)
    test_pred_hmm_gp.append(pred)
    test_std_hmm_gp.append(std)
    
test_std_hmm_gp = np.asarray(test_std_hmm_gp).flatten()
test_pred_hmm_gp = np.asarray(test_pred_hmm_gp).flatten()

# plot true vs predicted on the test_Y data
plot_y([test_Y, test_pred_hmm_gp],  'True values vs. model predictions on unseen (test_X) data', 'Hours', 'Electric load (normalized)', ['Truth','Prediction'])
in_ci = plot_CI('Confidence Interval on unseen (test_X) data', test_Y, test_pred_hmm_gp, test_std_hmm_gp, 95, 200, 325)
print 'The percent of true values that fall within the confidence interval is ', 100*in_ci,'%'