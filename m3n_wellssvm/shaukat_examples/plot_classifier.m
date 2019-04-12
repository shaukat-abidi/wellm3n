function plot_classifier( x,y,w,b )
m = size(x,1);
n = size(x,2);

pos_samples=[];
neg_samples=[];

for i=1:m
    if (y(i) == 1)
            pos_samples = [pos_samples;x(i,:)];
    else
            neg_samples = [neg_samples;x(i,:)];
    end
end

% draw line
lower_x1 = min(x(:,1));
upper_x1 = max(x(:,1));

% generate 100 points 
x1 = linspace(lower_x1,upper_x1,100);
x1 = x1';
slope = (-1 * w(1))/(w(2));
y1 = (slope * x1) + b;

plot(pos_samples(:,1),pos_samples(:,2),'r*');
hold on
plot(neg_samples(:,1),neg_samples(:,2),'b*');
plot(x1,y1,'g*');
hold off

end

