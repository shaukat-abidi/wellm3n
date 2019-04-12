clear all
close all
load('2D_data_hsvm.mat')
margin = 1;
m = size(samples,1);
n = size(samples,2);
x = samples;
e = ones(m,1);
kernel_matrix_y = (y*y').*(x * x');
clc
cvx_begin
variable alphas(m)
maximize( (alphas'*e) - (0.5 * alphas' * kernel_matrix_y * alphas) )
subject to
    alphas'*y == 0;
    alphas >= 0;
cvx_end

% evaluate w using representer theorem
w = alphas(1) * y(1) * x(1,:);
for i=2:m
    w_temp = alphas(i) * y(i) * x(i,:);
    w = w + w_temp;
end

w = w'; % column vector 
% find b
min_pos_score = x(1,:) * w;
max_neg_score = x(505,:) * w;
for i=1:m
    temp_score = x(i,:) * w;
    if(y(i) == 1)
        if (temp_score < min_pos_score)
            min_pos_score = temp_score;
        end
    else
        if (temp_score > max_neg_score)
            max_neg_score = temp_score;
        end
    end
end

b = -0.5*(max_neg_score+min_pos_score);

% plot hard margin svm
plot_classifier(x,y,w,b)