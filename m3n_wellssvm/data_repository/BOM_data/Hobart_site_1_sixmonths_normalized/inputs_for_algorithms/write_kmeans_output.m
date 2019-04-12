clear X Y

X = kmeans_data;
Y = inferred_state;
fid = fopen('kmeans_init_7.txt','w');
qid = 1;

for iter_rows=1:size(X,1)
    
    fprintf(fid,'%d qid:1',Y(iter_rows) );
    
    for iter_cols=1:size(X,2)
        fprintf(fid,' %d:%f',iter_cols, X(iter_rows,iter_cols) );        
    end
    
    fprintf(fid,'\n');

end
fclose(fid);
