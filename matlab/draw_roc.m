function [ output_args ] = draw_roc( path )
%DRAW_ROC Summary of this function goes here
%   Detailed explanation goes here

load('/home/tdteach/data/Megaface_Labels/val_lb.mat');

db = lmdb.DB(path,'MAPSIZE',1024^3);

n = 1000;

fea = zeros(n,256);

for i = 1:n
    s = sprintf('%010d',i-1);
    a = db.get(s);
    fea(i,:) = caffe_pb.fromDatum(a);
    fea(i,:) = fea(i,:)./norm(fea(i,:));
end

opts = zeros(n*(n-1)/2,1);
tgts = zeros(size(opts));
k = 0;
for i=1:n-1
    for j = i+1:n
      k = k+1;
      opts(k) = fea(i,:)*fea(j,:)';
      tgts(k) = (val_lb(i) == val_lb(j));
    end
end

clear db;

save([path,'/rst.mat'],'opts','tgts');

plotroc(tgts', opts');
end

