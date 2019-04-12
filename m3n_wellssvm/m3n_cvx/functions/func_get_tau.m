function [ tau ] = func_get_tau( sample, tau_fix, C )

tot_ex = size(sample.tot_ex);
tau=[];
tot_entries = 0; 

for iter_ex=1:tot_ex
    lot = sample.examples(iter_ex).lot;
    tot_entries = tot_entries + ( (2*lot) + (4*(lot-1)) );
    x = sample.examples(iter_ex).x;
    h_matrix = func_get_h_matrix(x,C);
    tau_imputed = h_matrix * tau_fix;
    tau=[tau;tau_imputed];
end

tau = -1 * tau;
assert(length(tau) == tot_entries);

end

