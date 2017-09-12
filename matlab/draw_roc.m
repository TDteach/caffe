function [ output_args ] = draw_roc(db_path, list_path )
%DRAW_ROC Summary of this function goes here
%   Detailed explanation goes here

lb = zeros(2,1);
i = 0;
fid = fopen(list_path,'r');
while ~feof(fid)
    s = fscanf(fid,'%s',1);
    if feof(fid)
        break;
    end
    i = i+1;
    lb(i,1) = fscanf(fid,'%d',1);
end
fclose(fid);

db = lmdb.DB(db_path,'MAPSIZE',1024^3);

fea = zeros(2,256);
i = 0;
cs = db.cursor('RDONLY',true);
while cs.next()
    a = cs.value;
    i = i+1;
    fea(i,:) = caffe_pb.fromDatum(a);
    fea(i,:) = fea(i,:)./norm(fea(i,:));
end
clear cs;

n = size(fea,1);

opts = zeros(n*(n-1)/2,1);
tgts = zeros(size(opts));
k = 0;
for i=1:n-1
    for j = i+1:n
      k = k+1;
      opts(k) = fea(i,:)*fea(j,:)';
      tgts(k) = (lb(i) == lb(j));
    end
end

clear db;

save([db_path,'/rst.mat'],'opts','tgts');

plotroc(tgts', opts');
end

