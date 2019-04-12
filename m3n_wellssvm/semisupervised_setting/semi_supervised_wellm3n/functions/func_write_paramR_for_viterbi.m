function func_write_paramR_for_viterbi( paramR, lot )
% for one example only
tot_entries = ( (2*lot) + 4*(lot-1));
assert(length(paramR) == tot_entries );

filename_paramR = '/home/ssabidi/Shaukat/m3n_wellssvm/semisupervised_setting/semi_supervised_wellm3n/params/paramR.txt';
fid = fopen(filename_paramR,'w');
for iter_entries=1:tot_entries
    fprintf(fid,'%f\n',paramR(iter_entries));
end
fclose(fid);


filename_lot = '/home/ssabidi/Shaukat/m3n_wellssvm/semisupervised_setting/semi_supervised_wellm3n/params/lot.txt';
fid = fopen(filename_lot,'w');
fprintf(fid,'%d\n',lot);
fclose(fid);

end

