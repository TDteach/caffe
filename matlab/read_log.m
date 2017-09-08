function [ loss ] = read_log( logfile, unit)
%READ_LOG Summary of this function goes here
%   Detailed explanation goes here

if ~exist('unit','var')
    unit = 10;
end
unit = floor(unit/10);

cmd = ['grep -i ", loss =" ', logfile, ' | cut -d= -f2 > tmp'];
system(cmd);
f = fopen('tmp','r');
loss = fscanf(f, '%f',[1000000,1]);
fclose(f);

n = size(loss,1);
m = n;
if (unit > 10)
    m = floor(n/unit);
    a = reshape(loss(1:unit*m), [unit, m]);
    a = mean(a);
    if (unit*m < n)
        a(m+1) = mean(loss(unit*m+1:n));
        m=m+1;
    end
else
    a = loss;
end

figure;
plot(a);
xlabel([num2str(unit*10), ' turns']);
% axis([1,m,0,15]);

loss = a;

end

