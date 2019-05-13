clc;clear;close all;


%% forward + backward test of mexSphericalConvolution.


%---------------------------------------------------------
%                                    test data preparation
%---------------------------------------------------------
rng(100);
ptCloud = load('bathtub.mat');
points = ptCloud.bathtub;
batch_size = 16;
batchData = repmat(reshape(points,[1,size(points)]),[batch_size,1,1]);

%---------------------------------------------------------
%  octree, spherical kernel and the network configuration
%---------------------------------------------------------
treeDepth = 6; binCapacity = 8;  
Nfilt = 8*2*3+1;  %[8,2,3] is the kernel size
f =  1/40; 
featSize = [3 32 64 64 64 128 128];



[dataBatch, mapBatch] = getOctreeBatch(batchData, treeDepth, binCapacity);

net(treeDepth) = struct('input',[],'filter',[],'bias',[],'map',[],...
                   'derOutput',[],'derFilter',[],'derBias',[]);           
net(1).input = dataBatch;  
for k = 1:treeDepth
    net(k).filter =f*randn(1,featSize(k),featSize(k+1),Nfilt,'single');
    net(k).bias = f*randn(1,featSize(k+1),'single');
    net(k).map = mapBatch{k};
end
net(end).derOutput = randn(1, 1, featSize(end), net(end).map(3,end)+1, 'single');

%---------------------------------------------------------
%                                 forward propagation test
%---------------------------------------------------------
gpuDevice; % reset gpuDevice

fprintf('----begin of forward test----\n\n');
size(net(1).input)
for layer = 1:treeDepth       
    input = gpuArray(net(layer).input);
    filter = gpuArray(net(layer).filter);
    bias = gpuArray(net(layer).bias);
    tic
    output = mexSphericalConvolution(input, filter, bias, net(layer).map);
    mexTime = toc;
    
    if(layer<treeDepth)                
        net(layer+1).input = output;        
    end
    size(output)
end
fprintf('\n----end of forward test----\n\n\n\n');



%---------------------------------------------------------
%                                backward propagation test
%---------------------------------------------------------
fprintf('-------begin of backward test-------\n\n');
size(net(end).derOutput)
for layer = treeDepth:-1:1         
    input = gpuArray(net(layer).input);
    filter = gpuArray(net(layer).filter);
    derOutput = gpuArray(net(layer).derOutput);
    tic
    [derInput, derFilter, derBias] = mexSphericalConvolution(input, ...
                                       filter, ...
                                       [], ...
                                       net(layer).map, derOutput);
    mexTime = toc;

    if layer>1
        net(layer-1).derOutput = derInput;
    end
    size(derInput)
end
fprintf('\n-------end of backward test-------\n');
