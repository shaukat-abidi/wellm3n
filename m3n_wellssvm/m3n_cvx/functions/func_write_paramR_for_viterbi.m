function func_write_paramR_for_viterbi( paramR, lot )
% for one example only
tot_entries = ( (2*lot) + 4*(lot-1));
assert(length(paramR) == tot_entries );

filename_paramR = '/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/paramR.txt';
fid = fopen(filename_paramR,'w');
for iter_entries=1:tot_entries
    fprintf(fid,'%f\n',paramR(iter_entries));
end
fclose(fid);


filename_lot = '/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/lot.txt';
fid = fopen(filename_lot,'w');
fprintf(fid,'%d\n',lot);
fclose(fid);

end

