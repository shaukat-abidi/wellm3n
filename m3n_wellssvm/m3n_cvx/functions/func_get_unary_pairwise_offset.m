function [ offset_unary, offset_pairwise ] = func_get_unary_pairwise_offset( sample )
% Offset for pairwise constraints
offset_pairwise=zeros(sample.tot_ex,1);
end_ind = 0;

for iter_ex=1:sample.tot_ex
    start_ind=end_ind;
    offset_pairwise(iter_ex) = start_ind;
    end_ind=start_ind+4*(sample.examples(iter_ex).lot-1);
end

% Offset for unary constraints
offset_unary = zeros(sample.tot_ex,1);
end_ind = 0;

for iter_ex=1:sample.tot_ex
    start_ind=end_ind;
    offset_unary(iter_ex) = start_ind;
    end_ind=start_ind+2*(sample.examples(iter_ex).lot);
end
end

