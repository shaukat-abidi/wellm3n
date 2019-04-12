function [ paramR ] = func_get_paramR( H,tau,DELTA,y_best )


paramR = (H*y_best) + tau + DELTA;


end

