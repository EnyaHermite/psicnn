function varargout = vl_sphconv(input, weights, biases, map, opts, cudnn, varargin)


if nargin==6 % forward propagation
    map = sortrows(map, 1); % In default, input is arranged in ascending IDs
    X = input; %size(X) = (1, 1, num*c_in)

%     [max(map(:,2)) length(weights)]
    weights = weights(map(:,2));
    F = cat(4,weights{:});
    
    if iscell(biases)
        c_out = numel(biases{1});
        biases = biases(map(:,2));
        B = cat(4,biases{:});
    else
        c_out = numel(biases);
        B = [];
    end
    
%     X = reshape(X',1,1,[]);
    output = vl_nnconv(X, F, B, ...
                      'pad', 0, ...
                      'stride', 1, ...
                      'dilate', 1, ...
                      opts{:}, ...
                      cudnn{:}) ; % fully connected convolution mode    
    
    output = reshape(output, c_out, [])';
    cluster = map(:,3);
    output = cellfun(@(x)avgpool(cluster, output, x), num2cell(unique(cluster)), 'uni', false); % output is arranged in ascending IDs as input
    
    output = cell2mat(output);
    output = bsxfun(@plus, output, biases(:)');
    output = reshape(output',1,1,[]);
%     assert(nargout==1, 'too many outputs!');
    varargout{1} = output;

elseif nargin==7 % backward propagation
    map = sortrows(map, 3); % In default, input is arranged in ascending IDs
    [dummy, dummy, Index] = unique(map(:,3));
    delta = varargin{1};
    c_in = size(weights{1},3);
    
        
    weights_used = weights(map(:,2));
    weights_used = cellfun(@(x)permute(x, [1 2 4 3]), weights_used, 'uni', false);
    F = cat(4, weights_used{:});
    
    c_out = numel(biases);
    delta = reshape(delta,c_out,[])';
    dzdw2 = sum(delta, 1); % need to re-check for correctness: should be correct(1st check).
    delta = delta(Index,:); % the error in layer l+1, note input is in layer l
    
    
%     assert(numel(delta)==c_out*size(map,1), 'dimension mismatch.');
%     assert(size(F,3)==c_out, 'dimension mismatch.');
%     assert(nargout==3, 'Three outputs required.');
    
    tab = tabulate(Index);
    N = tab(Index, 2);
    delta_unpool = bsxfun(@rdivide, delta, N);
    delta_unpool = reshape(delta_unpool',1,1,[]);
    dzdx = vl_nnconv(delta_unpool, F, [], ...
                    'pad', 0, ...
                    'stride', 1, ...
                    'dilate', 1, ...
                    opts{:}, ...
                    cudnn{:}); % fully connected convolution mode     
    
    delta_unpool = mat2cell(squeeze(delta_unpool), c_out*ones(1, size(map,1)));
    x = reshape(input,c_in,[])';
    x = mat2cell(x, ones(1, size(map,1)), c_in);
    dzdw1 = cellfun(@times, delta_unpool, x, 'uni', false);
    dzdw1 = cat(3, dzdw1{:});
    C = unique(map(:,2));
    dzdw1_used = cellfun(@(x)compute_dzdw1(map(:,2), dzdw1, x), num2cell(C), 'uni', false);
    dzdw1 = cell(size(weights));
    dzdw1(1:end) = {zeros(c_in,c_out)};
    dzdw1_used = cellfun(@(x)permute(x,[2 1]), dzdw1_used, 'uni', false);
    dzdw1(C) = dzdw1_used;
    
    varargout{1} = dzdx;
    varargout{2} = dzdw1;
    varargout{3} = dzdw2;           
end
end

function output = avgpool(cluster, input, idx)

input = input(cluster==idx,:);
output = mean(input, 1);
%output = output(:)'; % force output to be a row vector
end


function dzdw1 = compute_dzdw1(cluster, input, idx)

input = input(:, :, cluster==idx);
dzdw1 = sum(input, 3);
dzdw1 = squeeze(dzdw1); % force output to be a row vector
end
