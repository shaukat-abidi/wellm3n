function func_write_gt_file( base_path, label, L )

file_id=L-1;
filename_gt = strcat(base_path,'gt_',num2str(file_id),'.txt');
fid = fopen(filename_gt,'w');
tot_entries = length(label);

for iter_entries=1:tot_entries
    fprintf(fid,'%d\n',label(iter_entries));
end
fclose(fid);



end

