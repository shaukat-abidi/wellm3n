clear all
close all
clc
fname='/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/gt_0.txt';
sample = func_read_file( fname );
%sample.examples.x = 100 * sample.examples.x;

delta = func_get_delta_loss( sample.examples.y );
delta_psi = func_get_delta_psi_vec(sample.examples.x,sample.examples.y);
% H_psi = delta_psi*delta_psi';

T = sample.examples.lot;
unary_arg = 2*T;
pairwise_arg = 4*(T-1);  
C=0.1;

% Get pairwise constraints
[node_marginals, pairwise_summations] = func_get_pairwise_constraints(sample);
tot_pairwise_constraints = size(node_marginals,1);
%maximize ( (lambda_unary' * delta) - (0.5 * lambda_pairwise' * H_psi * lambda_pairwise) )
%maximize ( (lambda_unary' * delta) - (0.5 * quad_form(lambda_pairwise,H_psi) ) )
%maximize ( (lambda_unary' * delta) - (0.5 * square( norm(delta_psi' * lambda_pairwise) ) ) )
%maximize ( (lambda_unary' * delta) - (0.5 * norm(sqrtm(H_psi)*lambda_pairwise) ) )
cvx_begin
variable lambda_unary(unary_arg) 
variable lambda_pairwise(pairwise_arg)
maximize ( (lambda_unary' * delta) - (0.5 * square_pos( norm(delta_psi' * lambda_pairwise) ) ) )
subject to
for iter_const=1:tot_pairwise_constraints
    %(lambda_pairwise(pairwise_summations(iter_const,1)) + lambda_pairwise(pairwise_summations(iter_const,2))) == lambda_unary(node_marginals(iter_const));
    (lambda_pairwise(pairwise_summations(iter_const,1)) + lambda_pairwise(pairwise_summations(iter_const,2))) >= lambda_unary(node_marginals(iter_const));
    (lambda_pairwise(pairwise_summations(iter_const,1)) + lambda_pairwise(pairwise_summations(iter_const,2))) <= lambda_unary(node_marginals(iter_const));
end
lambda_pairwise >= 0;
for iter_tok=1:T
    %lambda_unary(2*iter_tok - 1) + lambda_unary(2*iter_tok) == C;
    lambda_unary(2*iter_tok - 1) + lambda_unary(2*iter_tok) >= C;
    lambda_unary(2*iter_tok - 1) + lambda_unary(2*iter_tok) <= C;
end
cvx_end

% Check if H is PSD
% eig_psi = eig(H_psi); 
%psi_symmetric = issymmetric(H_psi);
%[~,p] = chol(H_psi);
