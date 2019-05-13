clc;clear;close all;

f = 1/100;

sx = [-0.5 0.5];
sy = [-0.5 0.5];
sz = [-0.5 0.5];
[x,y,z] = meshgrid(sx,sy,sz);
points = single([x(:),y(:),z(:)]) + 0.1*rand(8,3,'single');
points = points(randperm(size(points,1)),:);
[points, map, demap] = mexOctreeMap(points, 1, 1, 'childAverage');

%% deconvolution:: forward
net(1).input = rand(1,1,3,1,'single');
net(1).filter = f*randn(1,3,16,73,'single');
net(1).bias = zeros(1,16,'single');
net(1).map = demap{1};

Nout = net(1).map(3,end) + 1;
out = zeros(16,Nout,'single');
input = squeeze(net(1).input);
for i = 1:size(points,1)
   chID =  net(1).map(1,i) + 1;
   filtID = net(1).map(2,i) + 1;
   paID = net(1).map(3,i) + 1;
   
   filter = net(1).filter(:,:,:,filtID);
   filter = squeeze(filter);
   out(:,paID) = filter'*input(:,chID);
end

input_ = gpuArray(net(1).input);
filter_ = gpuArray(net(1).filter);
bias_ = gpuArray(net(1).bias);
[mexOutput] = mexSphericalConvolution(input_, filter_, bias_, net(1).map);
mexOutput = gather(mexOutput);
mexOutput = squeeze(mexOutput);
res = abs(out-mexOutput);
[max(abs(out(:))) max(abs(mexOutput(:))) max(res(:))]


%% deconvolution:: backward
net(1).derOutput = rand(1,1,16,8,'single');

Nout = net(1).map(3,end) + 1;
out = zeros(16,Nout,'single');
input = squeeze(net(1).input);
derOutput = squeeze(net(1).derOutput);

derInput = zeros(3,1,'single');
derFilter = zeros(3,16,73,'single');
derBias = sum(derOutput,2)';
for i = 1:size(points,1)
   chID =  net(1).map(1,i) + 1;
   filtID = net(1).map(2,i) + 1;
   paID = net(1).map(3,i) + 1;
   Sz = net(1).map(4,i);
   
   filter = net(1).filter(:,:,:,filtID);
   filter = squeeze(filter);
   derInput(:,chID) = derInput(:,chID) + filter*derOutput(:,paID);
   
   derFilter(:,:,filtID) = derFilter(:,:,filtID) + input(:,chID)*derOutput(:,paID)';
end

input_ = gpuArray(net(1).input);
filter_ = gpuArray(net(1).filter);
bias_ = gpuArray(net(1).bias);
derOutput_ = gpuArray(net(1).derOutput);
[mexderInput, mexderFilter, mexderBias] = mexSphericalConvolution(input_, ...
                                          filter_, ...
                                          [], ...
                                          net(1).map, derOutput_);
mexderInput = gather(mexderInput);
mexderInput = squeeze(mexderInput);
mexderFilter = gather(mexderFilter);
mexderFilter = squeeze(mexderFilter);
mexderBias = gather(mexderBias);
mexderBias = squeeze(mexderBias);

res1 = abs(derInput-mexderInput);
res2 = abs(derFilter-mexderFilter);
res3 = abs(derBias-mexderBias);
[max(abs(derInput(:))) max(abs(mexderInput(:))) max(res1(:))]
[max(abs(derFilter(:))) max(abs(mexderFilter(:))) max(res2(:))]
[max(abs(derBias(:))) max(abs(mexderBias(:))) max(res3(:))]