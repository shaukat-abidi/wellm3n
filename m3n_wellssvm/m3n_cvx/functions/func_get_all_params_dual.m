function [H, tau, DELTA] = func_get_all_params_dual(sample, lambda_unary, lambda_pairwise, C)

%
% H,tau and DELTA will be returned from their function after proper
% scaling. Do not scale these variables from outside this function 
%
assert(size(sample,1) == 1);
assert(size(sample,2) == 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% JUST for one example
%%%%%%%%%%%%%%%%%%%%%%%%%%%
lot = sample.examples(1).lot;
x = sample.examples(1).x;

tau_fix  = func_get_tau_fix( sample, lambda_unary, lambda_pairwise, C );
tau = func_get_tau(sample, tau_fix, C);

% Get Delta (JUST FOR 1 example)
DELTA = func_get_delta_dual( lambda_unary,lot );

% Get H (one example only)
h_matrix = func_get_h_matrix(x,C);
H = h_matrix * h_matrix';

% Multiply H by 0.5
H = 0.5 * H;

end

