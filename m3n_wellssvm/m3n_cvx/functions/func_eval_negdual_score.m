function [ score ] = func_eval_negdual_score( H,tau,DELTA,y_binarized )

score = 0;

r = (H*y_binarized) + tau + DELTA;

score = y_binarized' * r;

assert(isscalar(score) == 1);

end

