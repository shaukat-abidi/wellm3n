function [ hj_cd ] = func_get_hj_cd_matrix( x,node_c,node_d,C )

% Returns hj_cd of the following dimension
% hj_cd = (6) x (2D+4)
D = size(x,2);
hj_cd=[];
trans_block = [zeros(1,D),zeros(1,D),C,0,0,0;
    zeros(1,D),zeros(1,D),0,C,0,0;
    zeros(1,D),zeros(1,D),0,0,C,0;
    zeros(1,D),zeros(1,D),0,0,0,C];

x_i=x(node_d,:);
pos_emission=[C*x_i,zeros(1,D),zeros(1,4)];
neg_emission=[zeros(1,D),C*x_i,zeros(1,4)];
hj_cd=[hj_cd;pos_emission]; %positive emission
hj_cd=[hj_cd;neg_emission]; %negative emission

hj_cd=[hj_cd;trans_block]; %transition block

assert(size(hj_cd,1) == 6);
assert(size(hj_cd,2) == (2*D + 4));


end

