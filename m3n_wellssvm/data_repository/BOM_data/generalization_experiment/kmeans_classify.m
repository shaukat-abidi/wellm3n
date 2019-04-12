function [ label_vec ] = kmeans_classify( centroid, kmeans_data )
label_vec = [];
% ex = 1 x dim
% centroid = No. of classes x dim of ex
% for BOM Dataset : centroid = 2 x 15

% THIS IMPLEMENTATION IS FOR BOM DATASET
assert(size(centroid,1) == 2)
assert(size(centroid,2) == 15)

for iter_ex=1:size(kmeans_data,1)    
    dist_c1 = pdist2(centroid(1,:),kmeans_data(iter_ex,:));
    dist_c2 = pdist2(centroid(2,:),kmeans_data(iter_ex,:));
    
    %fprintf('%f %f\n', dist_c1, dist_c2);
    
    % assign label to the closest centroid
    if (dist_c1<=dist_c2)
        label_vec = [label_vec;1];
    else
        label_vec = [label_vec;2];
    end
    
end



end

