function [xBatch, mapBatch] = getOctreeBatch(batchData, treeDepth, binCapacity)
                                   
batchSize = size(batchData,1);
xBatch = cell(batchSize,1); 
mapBatch = cell(1,treeDepth); 

for i = 1:batchSize    
    points = squeeze(batchData(i,:,:)); % size (N,3)
    
    points = points - mean(points,1); % remove translation    
    scale = max(abs(points(:)));         
    points = points/scale; 

    [points, map] = mexOctreeMap(points, treeDepth, binCapacity, 'childAverage');   
    
    x = points';    
    xBatch{i} = x;    
    for l = 1:treeDepth
        if i>1
            start1 = mapBatch{l}(1,end) + 1;
            start3 = mapBatch{l}(3,end) + 1;
            map{l}(1,:) = map{l}(1,:) + start1;
            map{l}(3,:) = map{l}(3,:) + start3;
        end
        mapBatch{l} = cat(2, mapBatch{l}, map{l});
    end
end

xBatch = cat(2, xBatch{:});
xBatch = reshape(xBatch,[1 1 size(xBatch)]);
xBatch(isnan(xBatch)) = 0;
xBatch = gpuArray(xBatch);
