function [ lambda_unary, lambda_pairwise ] = func_run_cvx_single_labeling( sample, C )

delta = func_get_delta_loss( sample.examples.y );
delta_psi = func_get_delta_psi_vec(sample.examples.x,sample.examples.y);
H_psi = delta_psi*delta_psi';


%%%%%%%%%%%%%%%%%%%%%%%%%%
% For SINGLE SEQUENCE ONLY
%%%%%%%%%%%%%%%%%%%%%%%%%%
T = sample.examples.lot;
unary_arg = 2*T;
pairwise_arg = 4*(T-1);  

% Get pairwise constraints
[node_marginals, pairwise_summations] = func_get_pairwise_constraints(sample);
tot_pairwise_constraints = size(node_marginals,1);

cvx_begin
variable lambda_unary(unary_arg) 
variable lambda_pairwise(pairwise_arg)
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



end

