function [ mu_em, sigma_em, mixmat0, prior0, transmat0 ] = func_get_mu_sigma( filename )

data_labels = dlmread(filename);
labels = data_labels(:,1);
qid = data_labels(:,2);
data_feats = data_labels(:,3:end);

lot = size(data_labels,1);
pos_data=[];
neg_data=[];

for iter_tok=1:lot
    if(labels(iter_tok) == 1)
        pos_data = [pos_data;data_feats(iter_tok,:)];
    else
        neg_data = [neg_data;data_feats(iter_tok,:)];
    end        
end

assert(lot == (size(pos_data,1) + size(neg_data,1)));
pos_mean = mean(pos_data);
neg_mean = mean(neg_data);
sigma_pos = cov(pos_data);
sigma_neg = cov(neg_data);

mu_em = [pos_mean',neg_mean'];
sigma_em(:,:,1) = sigma_pos;
sigma_em(:,:,2) = sigma_neg;

tot_pos_tok = size(pos_data,1);
tot_neg_tok = size(neg_data,1);

mixmat0 = [0.5;0.5];
prior0 = [tot_pos_tok/lot;tot_neg_tok/lot];

tok_pp=0;
tok_pn=0;
tok_np=0;
tok_nn=0;

for iter_tok=2:lot
    if(labels(iter_tok-1) == 1 && labels(iter_tok) == 1)
        tok_pp = tok_pp + 1;
    end
    if(labels(iter_tok-1) == 1 && labels(iter_tok) == 2)
        tok_pn = tok_pn + 1;
    end
    if(labels(iter_tok-1) == 2 && labels(iter_tok) == 1)
        tok_np = tok_np + 1;
    end
    if(labels(iter_tok-1) == 2 && labels(iter_tok) == 2)
        tok_nn = tok_nn + 1;
    end
end

transmat0=[tok_pp tok_pn;
    tok_np tok_nn];

assert (lot == (tok_pp+tok_pn+tok_np+tok_nn+1) );

transmat0=[tok_pp/tot_pos_tok tok_pn/tot_pos_tok;
    tok_np/tot_neg_tok tok_nn/tot_neg_tok];

% Normalize
transmat0(1,1) = transmat0(1,1) / ( transmat0(1,1) + transmat0(1,2) );   
transmat0(1,2) = transmat0(1,2) / ( transmat0(1,1) + transmat0(1,2) );   

transmat0(2,1) = transmat0(2,1) / ( transmat0(2,1) + transmat0(2,2) );
transmat0(2,2) = transmat0(2,2) / ( transmat0(2,1) + transmat0(2,2) );   



end