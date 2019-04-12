clear all
close all
clc
fname='data_synthetic_10_frames.txt';
sample = func_read_file( fname );

L = 2;
mu = zeros(L,1);


mu(1) = 0.5;
mu(2) = 0.5;

delta = func_get_delta( sample.examples.y );
delta_psi = func_get_delta_psi_vec(sample.examples.x,sample.examples.y);
%H_psi = delta_psi*delta_psi';

delta_buffer = delta;
delta_psi_buffer = delta_psi;

delta = (mu(1)*delta_buffer) + (mu(2)*delta_buffer);
delta_psi = (sqrt(mu(1))*delta_psi_buffer) + (sqrt(mu(2))*delta_psi_buffer);
H_psi = delta_psi*delta_psi';

% Quick check
%delta = [delta;delta];

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVX Optimization Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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