function [ sample ] = func_read_file( fname )

data = dlmread(fname,' ');
tot_rows=size(data,1);
tot_cols=size(data,2);

% data: 1st col = class_label |  2nd col = qid | 3rd col - end = features
tot_feats = tot_cols-2; %Fixed
tot_ex = data(tot_rows,2); %Read last line of fname


sample = struct('tot_ex',tot_ex,'examples',repmat(struct('x',[],'y',[],'lot',0),tot_ex,1) );

for iter_rows=1:tot_rows
    current_example = data(iter_rows,2); %2nd column of every row is the qid
    current_label = data(iter_rows,1); %1st column of every row is token label 
    feats = data(iter_rows,3:end);
    
    sample.examples(current_example).x = [sample.examples(current_example).x;feats];
    sample.examples(current_example).y = [sample.examples(current_example).y;current_label];
    sample.examples(current_example).lot = sample.examples(current_example).lot + 1;
end

% check if everything is fine
for iter_ex=1:tot_ex
    assert( size(sample.examples(iter_ex).x,2) == tot_feats);
end

end