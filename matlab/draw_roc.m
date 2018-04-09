function [tpr, fpr] = draw_roc(db_path, list_path, dim, varargin )
%DRAW_ROC Summary of this function goes here
%   Detailed explanation goes here

% db_path = '/home/tdteach/workspace/caffe_model/deep-residual-networks/ff';
% l_path = '/home/tdteach/data/Megaface_Labels/lists/list_caffe_val.txt';

system(['wc -l ',list_path,' >tmp']);
fid = fopen('tmp','r');
n = fscanf(fid,'%d',1);
fclose(fid);
system('rm -rf tmp');

lb = zeros(n,1);
img_path = cell(n,1);
fid = fopen(list_path,'r');
for i = 1:n
    s = fscanf(fid,'%s',1);
    img_path{i,1} = s;
    lb(i,1) = fscanf(fid,'%d',1);
end
fclose(fid);

db = lmdb.DB(db_path,'MAPSIZE',1024^3);

fea = zeros(n,dim);
i = 0;
cs = db.cursor('RDONLY',true);
while cs.next()
    i = i+1;
    %the same way to use the readdatum function
    %datum = readdatum(cs.value);
    %fea(i,:) = datum.float_data(:);
    a = cs.value;
    fea(i,:) = flip(caffe_pb.fromDatum(a));
    fea(i,:) = fea(i,:)./norm(fea(i,:));
end
clear cs;
n = i;
fea = fea(1:n,:);

m = n;
if nargin > 3
    m = int32(varargin{1});
end

coss = zeros(m,n);
opts = zeros(m*(n-1)/2,1);
tgts = zeros(size(opts));
k = 0;
for i=1:m
    for j = i+1:n
      k = k+1;
      opts(k) = fea(i,:)*fea(j,:)';
      coss(i,j) = opts(k);
      tgts(k) = (lb(i) == lb(j));
    end
end

clear db;

[tpr, fpr, thr] = roc(tgts', opts');

save([db_path,'/rst.mat'],'opts','tgts','coss','tpr','fpr','thr','img_path','fea');

plot(fpr,tpr);
xlabel('fpr');
ylabel('tpr');

end

