function [ y_binarized ] = func_get_binarized_labeling( labeling_y )
% returns binarized labeling for vector y
% 1 = 10
% 2 = 01
% 11 = 1000
% 12 = 0100
% 21 = 0010
% 22 = 0001

y_binarized = [];
pos_frame=[1;0];
neg_frame=[0;1];
t_pp=[1;0;0;0];
t_pn=[0;1;0;0];
t_np=[0;0;1;0];
t_nn=[0;0;0;1];
lot = length(labeling_y);

assert(size(labeling_y,1) == lot);

% encode frame label
for iter_tok=1:lot
    if (labeling_y(iter_tok) == 1)
        y_binarized = [y_binarized;pos_frame];
    else
        y_binarized = [y_binarized;neg_frame];        
    end
end

% encode transition labels
for iter_tok=2:lot
    
    past = iter_tok - 1;
    current = iter_tok;
    if(labeling_y(past) == 1 && labeling_y(current) == 1)
        y_binarized = [y_binarized;t_pp];
    elseif(labeling_y(past) == 1 && labeling_y(current) == 2)
        y_binarized = [y_binarized;t_pn];
    elseif(labeling_y(past) == 2 && labeling_y(current) == 1)
        y_binarized = [y_binarized;t_np];
    else
        y_binarized = [y_binarized;t_nn];
    end
    
end

assert(length(y_binarized) == (2*lot + 4*(lot-1)));

end

