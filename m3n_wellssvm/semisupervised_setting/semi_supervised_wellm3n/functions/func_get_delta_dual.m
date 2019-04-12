function [ DELTA ] = func_get_delta_dual( lambda_unary,lot )
% DELTA: linear term of m3n dual (DELTA^{T}y_gt)
% lambda_unary: vector with 2*lot entries
% lambda_unary = [lambda(u_a=1) lambda(u_a=2) ... lambda(u_lot=1) lambda(u_lot=2)]
assert(length(lambda_unary) == 2*lot);
trans_fill = [0;0;0;0];

% Populate DELTA
% DELTA = [lambda(u_a=2) lambda(u_a=1) ... lambda(u_lot=2) lambda(u_lot=1) 0000 ... 0000]
DELTA=[];

for iter_tok=1:lot
    first_entry = (2*iter_tok) - 1;
    second_entry = (2*iter_tok);
    DELTA=[DELTA;lambda_unary(second_entry);lambda_unary(first_entry)];
end

% FILL transition zeros
for iter_tok=2:lot
    DELTA=[DELTA;trans_fill];
end

% Scale Delta by -1
DELTA = -1 * DELTA;

assert(length(DELTA) == (2*lot + 4*(lot-1)));


end

