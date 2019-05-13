clc;clear;close all;

addpath('D:\treeNet\matconvnet\matlab');
vl_setupnn();
eps = 1e-2;

%% forward + backward test between vl_sphconv and mexSphericalConvolution.
%% This test covers only spherical convolution, the spherical deconvolution
%% will not be covered

%---------------------------------------------------------
%                                    test data preparation
%---------------------------------------------------------
rng(100);
ptCloud = pcread('bathtub.ply');
points = ptCloud.Location;
Level = 10; f =  1/40; binCapacity = 8; kNN = 10; 
featSize = [6 16 16 32 32 64 64 128 128 256 256];%[6 32 32 32 64 64 64 64 64 64 128 128 128 512 512 1024];
tic
[points, map] = mexOctreeMap(points, Level, binCapacity, 'childAverage');
toc
net(Level) = struct('input',[],'filter',[],'bias',[],'map',[],...
                   'derOutput',[],'derFilter',[],'derBias',[]);
normals = pcnormals(pointCloud(points), kNN);             
net(1).input = [points normals]';  
net(1).input = reshape(net(1).input, [1 1 size(net(1).input)]);
for k = 1:Level
    net(k).filter =f*randn(1,featSize(k),featSize(k+1),1+72*k,'single');
    net(k).bias = f*randn(1,featSize(k+1),'single');
    net(k).map = map{k};
end
net(end).derOutput = randn(1, 1, featSize(end), net(end).map(3,end)+1, 'single');

%---------------------------------------------------------
%                                 forward propagation test
%---------------------------------------------------------

gpuDevice; % reset gpuDevice

sumTime = [0, 0];
fprintf('----begin of forward test----\n\n');
for layer = 1:Level   
    [m,n,d,s] = size(net(layer).filter);
    weights = reshape(net(layer).filter, 1, 1, n, d*s);
    weights = mat2cell(weights, 1, 1, n, d*ones(1,s));
    input = reshape(net(layer).input,1,1,[]);
    map = net(layer).map(1:3,:)'+1;
    tic
    mOutput = vl_sphconv(input, weights, net(layer).bias, map, {}, {});
    mTime = toc;
    
    
    input_ = gpuArray(net(layer).input);
    filter_ = gpuArray(net(layer).filter);
    bias_ = gpuArray(net(layer).bias);
    tic
    [mexOutput] = mexSphericalConvolution(input_, filter_, bias_, net(layer).map);
    mexTime = toc;
    
    sumTime(1) = sumTime(1) + mTime;
    sumTime(2) = sumTime(2) + mexTime;
    fprintf('vl_sphconv: %.6f(s), mexSphconv: %.6f(s).\n', mTime, mexTime);
    
    output = reshape(mOutput,1,1,[],map(end,3));    
    output_ = gather(mexOutput);
    res = abs(output - output_);
    fprintf('layer %d:\n%.8f\n%.8f\n%.8f\n\n', layer, max(output(:)), max(output_(:)), max(res(:)));        
    if(layer<Level)                
        net(layer+1).input = output_;
    end
    
    % plot figure for debugging
    figure(layer),plot(output(:),'ro'),hold on
    plot(output_(:),'g+'),hold on  
end
fprintf('\nTime in total:\nvl_sphconv: %.6f(s), mexSphconv: %.6f(s).\n', sumTime(1), sumTime(2));
fprintf('\n----end of forward test----\n\n\n\n');



%---------------------------------------------------------
%                                backward propagation test
%---------------------------------------------------------
sumTime = [0, 0];
fprintf('-------begin of backward test-------\n\n');
for layer = Level:-1:1   
    [m,n,d,s] = size(net(layer).filter);
    weights = reshape(net(layer).filter, 1, 1, n, d*s);
    weights = mat2cell(weights, 1, 1, n, d*ones(1,s));
    input = reshape(net(layer).input,1,1,[]);
    map = net(layer).map(1:3,:)'+1;
    derOutput = reshape(net(layer).derOutput,1,1,[]);
    tic
    [mderInput, mderFilter, mderBias] = vl_sphconv(input, weights, net(layer).bias, map, {}, {}, derOutput);
    mTime = toc;
    
    
    input_ = gpuArray(net(layer).input);
    filter_ = gpuArray(net(layer).filter);
    derOutput_ = gpuArray(net(layer).derOutput);
    tic
    [mexderInput, mexderFilter, mexderBias] = mexSphericalConvolution(input_, ...
                                       filter_, ...
                                       [], ...
                                       net(layer).map, derOutput_);
    mexTime = toc;
    
    sumTime(1) = sumTime(1) + mTime;
    sumTime(2) = sumTime(2) + mexTime;
    fprintf('vl_sphconv: %.6f(s), mexSphconv: %.6f(s).\n', mTime, mexTime);
    
    derBias = mderBias; 
    derBias_ = gather(mexderBias);                               
    res1 = abs(derBias - derBias_);
    
    derInput = reshape(mderInput,1,1,[],map(end,1));
    derInput_ = gather(mexderInput);
    res2 = abs(derInput - derInput_);
    
    derFilter = cat(3, mderFilter{:});
    derFilter = reshape(derFilter,[m,n,d,s]);
    derFilter_ = gather(mexderFilter);
    res3 = abs(derFilter - derFilter_);

    fprintf('layer %d:\n%.8f,%.8f,%.8f\n%.8f,%.8f,%.8f\n%.8f,%.8f,%.8f\n\n', ...
           layer, max(derBias(:)), max(derInput(:)), max(derFilter(:)), ...
           max(derBias_(:)), max(derInput_(:)), max(derFilter_(:)), ...
           max(res1(:)), max(res2(:)), max(res3(:)));    
    if layer>1
        net(layer-1).derOutput = derInput;
    end
end
fprintf('\nTime in total:\nvl_sphconv: %.6f(s), mexSphconv: %.6f(s).\n', sumTime(1), sumTime(2));
fprintf('\n-------end of backward test-------\n');