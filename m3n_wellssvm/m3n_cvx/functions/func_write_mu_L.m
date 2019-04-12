function func_write_mu_L( mu, L )

filename_mu = '/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/mu.txt';
fid = fopen(filename_mu,'w');
for iter_entries=1:length(mu)
    fprintf(fid,'%f\n',mu(iter_entries));
end
fclose(fid);


filename_lot = '/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/L.txt';
fid = fopen(filename_lot,'w');
fprintf(fid,'%d\n',L);
fclose(fid);

end

