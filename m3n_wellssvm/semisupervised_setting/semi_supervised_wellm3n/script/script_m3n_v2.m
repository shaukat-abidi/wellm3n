clear all
close all
clc

addpath('/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/functions/');

fname='data_C3.txt';
sample = func_read_file( fname );

delta = func_get_delta_loss( sample.examples.y );
delta_psi = func_get_delta_psi_vec(sample.examples.x,sample.examples.y);
H_psi = delta_psi*delta_psi';


%%%%%%%%%%%%%%%%%%%%%%%%%%
% For SINGLE SEQUENCE ONLY
%%%%%%%%%%%%%%%%%%%%%%%%%%
T = sample.examples.lot;
unary_arg = 2*T;
pairwise_arg = 4*(T-1);  
C=10;

% Get pairwise constraints
[node_marginals, pairwise_summations] = func_get_pairwise_constraints(sample);
tot_pairwise_constraints = size(node_marginals,1);

cvx_begin
variables lambda_unary(unary_arg) lambda_pairwise(pairwise_arg)
maximize ( (lambda_unary' * delta) - (0.5 * lambda_pairwise' * H_psi * lambda_pairwise) )
subject to
for iter_const=1:tot_pairwise_constraints
    (lambda_pairwise(pairwise_summations(iter_const,1)) + lambda_pairwise(pairwise_summations(iter_const,2))) >= lambda_unary(node_marginals(iter_const));
    (lambda_pairwise(pairwise_summations(iter_const,1)) + lambda_pairwise(pairwise_summations(iter_const,2))) <= lambda_unary(node_marginals(iter_const));
end
lambda_pairwise >= 0;
for iter_tok=1:T
    lambda_unary(2*iter_tok - 1) + lambda_unary(2*iter_tok) >= C;
    lambda_unary(2*iter_tok - 1) + lambda_unary(2*iter_tok) <= C;
end
cvx_end

% Check if H is PSD
% eig_psi = eig(H_psi); 
% psi_symmetric = issymmetric(H_psi);
%[~,p] = chol(H_psi);