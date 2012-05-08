% setup
sigmoid = inline('.5 * (1 + tanh(z / 2.))');

XX = randn(2000, 100);     % all data
X = X(1:20, :);     % mini batch
W = randn(100, 200) * .03;
vBias = zeros(1, 300);
hBias = zeros(1, 200);

figure(1);
imagesc(sigmoid(X*W + repmat(hBias, 20, 1)), [0, 1]);
colormap('gray'); axis('equal');


function plotit(values)
    hist(values(:));
    title(sprintf('mm = %g', mean(mean(abs(values)))));
end

figure(2);
subplot(231); plotit(vBias);
subplot(232); plotit(W);
subplot(233); plotit(hBias);
subplot(234); plotit(vBias / 100);
subplot(235); plotit(W     / 100);
subplot(236); plotit(hBias / 100);



figure(3);
nRows = 5;
nCols = 5;
tiled = ones(11*nRows, 11*nCols) * .2  # dark gray borders

for row = 1:nRows
    for col = 1:nCols
        patch = W(:,(row-1)*nCols+col);
        normPatch = (patch - min(patch)) / (max(patch)-min(patch)+1e-6);
        tiled((row-1)*11+1:(row-1)*11+10, (col-1)*11+1:(col-1)*11+10) = reshape(normPatch, 10, 10);
    end
end
imagesc(tiled, [0, 1]);
colormap('gray'); axis('equal');
