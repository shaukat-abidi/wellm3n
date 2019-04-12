function [ h_matrix ] = func_get_h_matrix( x,C )
% Returns h_i of the following dimension
% h_i = (2T_i + 4(T_i-1)) x (2D+4)
lot = size(x,1);
D = size(x,2);
h_matrix=[];
trans_block = [zeros(1,D),zeros(1,D),C,0,0,0;
               zeros(1,D),zeros(1,D),0,C,0,0;
               zeros(1,D),zeros(1,D),0,0,C,0;
               zeros(1,D),zeros(1,D),0,0,0,C];

for iter_tok=1:lot
    x_i=x(iter_tok,:);
    pos_emission=[C*x_i,zeros(1,D),zeros(1,4)];
    neg_emission=[zeros(1,D),C*x_i,zeros(1,4)];
    h_matrix=[h_matrix;pos_emission]; %positive emission
    h_matrix=[h_matrix;neg_emission]; %negative emission
end

for iter_tok=1:(lot-1)
    h_matrix=[h_matrix;trans_block]; %transition block
end

assert(size(h_matrix,1) == (2*lot + 4*(lot-1)));
assert(size(h_matrix,2) == (2*D + 4));


end

