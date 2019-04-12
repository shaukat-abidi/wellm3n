function [ loss ] = func_loss( y_tok,u )
loss = 0;

if (y_tok ~= u)
    loss = 1;
end

end

