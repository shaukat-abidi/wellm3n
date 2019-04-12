function [ delta ] = func_get_delta_loss( y_gt )

lot = length(y_gt);
delta = zeros(2*lot,1);
iter_delta = 1;

for iter_tok=1:lot
    delta(iter_delta) = func_loss(y_gt(iter_tok), 1);
    delta(iter_delta+1) = func_loss(y_gt(iter_tok), 2);
    iter_delta = iter_delta+2;
end

assert (iter_delta == 2*lot + 1);
end

