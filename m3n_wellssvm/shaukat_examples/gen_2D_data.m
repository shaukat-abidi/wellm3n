close all 
clear all
clc
m_neg = 500;
m_pos = 500;
m = m_neg + m_pos; 
n = 2;

% positive samples
mu_pos = [5 5];
sigma_pos = [1 0; 0 1];
pos_samples = mvnrnd(mu_pos,sigma_pos,m_pos);
plot(pos_samples(:,1),pos_samples(:,2),'r*')
hold on

% negative samples
mu_neg = [-5 -5];
sigma_neg = [1 0; 0 1];
neg_samples = mvnrnd(mu_neg,sigma_neg,m_neg);
plot(neg_samples(:,1),neg_samples(:,2),'b*')

% aggregate samples and labels
y_pos = ones(m_pos,1);
y_neg = -1 * ones(m_neg,1);
y = [y_pos;y_neg];

samples = [pos_samples;neg_samples];
save('2D_data_hsvm.mat','samples','y');