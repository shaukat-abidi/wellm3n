% addpath(genpath('/home/sabidi/Shaukat/WellSSVM/implementation/EM/MurphyToolbox/HMMall'))

%true labels=1,false labels=2
close all
clear all
load('mat_files/data_labels.mat');

%Scale dataset by scale_factor
%scale_factor = 100;
%for i=1:size(data,1)
%    for j=1:size(data,2)
%        data(i,j) = data(i,j) * scale_factor;
%    end    
%end

X = data;
filtered_labels = gt_labels;
clear data gt_labels
clc
reduced_dim_data = X;
clear X

O = size(reduced_dim_data,2);    %Number of coefficients in a vector
T = size(reduced_dim_data,1);   %Number of vectors in a sequence
nex = 1; %Number of sequences
M = 1;     %Number of mixtures
Q = 2;    %Number of states
cov_type = 'full';

labels = filtered_labels;
% change label 0 to 2
%for iter_labels=1:length(labels)
%    if(labels(iter_labels) == 0)
%        labels(iter_labels) = 2; 
%    end
%end


% %data = randn(O,T,nex);
% data(1,:,:) = obs;
data(:,:,1) = reduced_dim_data';
% 
% initial guess of parameters
%prior0 = normalise(rand(Q,1));
%transmat0 = mk_stochastic(rand(Q,Q));

% [mu0, Sigma0] = mixgauss_init(Q*M, data, cov_type);
% function [mu, Sigma, weights] = mixgauss_init(M, data, cov_type, method)
% MIXGAUSS_INIT Initial parameter estimates for a mixture of Gaussians
% function [mu, Sigma, weights] = mixgauss_init(M, data, cov_type. method)
%
% INPUTS:
% data(:,t) is the t'th example
% M = num. mixture components
% cov_type = 'full', 'diag' or 'spherical'
% method = 'rnd' (choose centers randomly from data) or 'kmeans' (needs netlab)
%
% OUTPUTS:
% mu(:,k)
% Sigma(:,:,k)
% weights(k)

%mu0 = reshape(mu0, [O Q M]);
%Sigma0 = reshape(Sigma0, [O O Q M]);
[mu0, Sigma0, mixmat0, prior0, transmat0] = func_get_mu_sigma('/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/gt_0.txt');
%mixmat0 = mk_stochastic(rand(Q,M));

[LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    mhmm_em(data, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 200);

% LEARN_MHMM Compute the ML parameters of an HMM with (mixtures of) Gaussians output using EM.
% [ll_trace, prior, transmat, mu, sigma, mixmat] = learn_mhmm(data, ...
%   prior0, transmat0, mu0, sigma0, mixmat0, ...)
%
% Notation: Q(t) = hidden state, Y(t) = observation, M(t) = mixture variable
%
% INPUTS:
% data{ex}(:,t) or data(:,t,ex) if all sequences have the same length
% prior(i) = Pr(Q(1) = i),
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
% mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k ]
% Sigma(:,:,j,k) = Cov[Y(t) | Q(t)=j, M(t)=k]
% mixmat(j,k) = Pr(M(t)=k | Q(t)=j) : set to [] or ones(Q,1) if only one mixture component
%
% If the number of mixture components differs depending on Q, just set  the trailing
% entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
% then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.


loglik = mhmm_logprob(data, prior1, transmat1, mu1, Sigma1, mixmat1);
%Accumulate total-pos and tot-neg tokens
tot_pos_tokens=0;
tot_neg_tokens=0;
tot_tokens_in_dataset = 0;
for chosen_sample=1:nex
    gt_states = labels;
    tot_tok = length(gt_states);
    tot_tokens_in_dataset = tot_tokens_in_dataset + tot_tok;
    for iter_tok = 1:tot_tok
        if(gt_states(iter_tok) == 1)
            tot_pos_tokens = tot_pos_tokens + 1;
        else
            tot_neg_tokens = tot_neg_tokens + 1;
        end
            
    end
end

% decodes one the sequences with the learned model and compares with the generating sequence:

chosen_sample = 1; % arbitrary choice %
% token = 1;
% obs_data = data(:,token,chosen_sample);
% component_chosen = 1;
% state_chosen = 1;
% des_sig = Sigma1(:,:,component_chosen);
% des_mu = mu1(:,state_chosen);
% alpha = mixmat1(state_chosen,component_chosen);
% p_x = get_P(obs_data,des_mu,des_sig)

sum_true_predictions = 0;
tot_tokens = T;
tp=0;
tn=0;
fp=0;
fn=0;
tok_in_sample=0;
path_predicted=[];

for t = 1:T %no.of.tokens
    token = t;
    obs_data = data(:,token,chosen_sample);
    for q = 1:Q %no.of hidden states
        p_x = 0;
        obslik(q,t) = 0;
        for component_chosen=1:size(mixmat1,2) %1:total_mixture_comps
            state_chosen = q;
            des_sig = Sigma1(:,:,component_chosen);
            des_mu = mu1(:,state_chosen);
            alpha = mixmat1(state_chosen,component_chosen);
            p_x = alpha * get_P(obs_data,des_mu,des_sig);
            obslik(q,t) = obslik(q,t) + p_x;
        end
    end
end


%
% for t = 1:T %no.of.tokens
%     for q = 1:Q%no.of hidden states
%        obslik(q,t) = obsmat2(q,data(t,chosen_sample));
%     end
% end
%
path = viterbi_path(prior1, transmat1, obslik);
path_predicted = [path_predicted;path]; %Store predicted path
%
% accuracy (%):
path = path';
correct_prediction = sum((labels == path));
sum_true_predictions = sum_true_predictions + correct_prediction;

%Accumulate tp,tn,fp,fn
tok_in_sample = length(path);
for iter_tok=1:tok_in_sample
	if(path(iter_tok) == 1 && labels(iter_tok) == 1)
		tp = tp + 1;
	end
	if(path(iter_tok) == 1 && labels(iter_tok) == 2)
		fp = fp + 1;
	end
	if(path(iter_tok) == 2 && labels(iter_tok) == 1)
		fn = fn + 1;
	end
	if(path(iter_tok) == 2 && labels(iter_tok) == 2)
		tn = tn + 1;
	end
end


%accuracy = sum((labels == path))/T * 100
accuracy = sum_true_predictions / T * 100;
fprintf('tot-tokens = %d tot-pos-tokens = %d tot-neg-tokens = %d\n',tot_tokens_in_dataset,tot_pos_tokens,tot_neg_tokens);
fprintf('tp=%d,tn=%d,fp=%d,fn=%d\n',tp,tn,fp,fn);
fprintf('EM Accuracy = %f\n',accuracy);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
f1_score = (2*precision*recall)/(precision+recall);
fprintf('precision = %f , recall = %f , f1_score = %f \n\n\n',precision,recall,f1_score);

%%%% Calling EM again to invert result %%%%%%
if(accuracy>0)
    [rows,cols] = size(path);
    % Invert IDX
    for iter_rows=1:rows
        for iter_cols=1:cols
            if(path(iter_rows,iter_cols) == 1)
                path(iter_rows,iter_cols) = 2;
            else
                path(iter_rows,iter_cols) = 1;
            end
        end
    end
    
sum_true_predictions = 0;
tot_tokens = T;
tp=0;
tn=0;
fp=0;
fn=0;
tok_in_sample=0;

correct_prediction = sum((labels == path));
sum_true_predictions = sum_true_predictions + correct_prediction;

%Accumulate tp,tn,fp,fn
tok_in_sample = length(path);
for iter_tok=1:tok_in_sample
	if(path(iter_tok) == 1 && labels(iter_tok) == 1)
		tp = tp + 1;
	end
	if(path(iter_tok) == 1 && labels(iter_tok) == 2)
		fp = fp + 1;
	end
	if(path(iter_tok) == 2 && labels(iter_tok) == 1)
		fn = fn + 1;
	end
	if(path(iter_tok) == 2 && labels(iter_tok) == 2)
		tn = tn + 1;
	end
end


% % accuracy (%):
accuracy = sum_true_predictions / tot_tokens * 100;
fprintf('(Result Inverted)tot-tokens = %d tot-pos-tokens = %d tot-neg-tokens = %d\n',tot_tokens_in_dataset,tot_pos_tokens,tot_neg_tokens);
fprintf('(Result Inverted)tp=%d,tn=%d,fp=%d,fn=%d\n',tp,tn,fp,fn);
fprintf('(Result Inverted)EM Accuracy = %f\n',accuracy);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
f1_score = (2*precision*recall)/(precision+recall);
fprintf('(RESULT INVERTED)precision = %f , recall = %f , f1_score=%f \n',precision,recall,f1_score);
    
end


