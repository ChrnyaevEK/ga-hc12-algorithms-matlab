clear; clc;
clc; clf; clear;

lims = [-5 10];
maxGen = 1000;

% [x, y, historyBestY] = hc12(@rosen, 10, 4, lims, maxGen, 0.0005, 10);
% 
% figure(1);
% plot(historyBestY);
stats = zeros(100, 1);
statsTime = zeros(100, 1);

f = figure(1);

ax = subplot(1, 3, 1);
ax.YLim = [0 30];
title('Cost progress');
xlabel('Epoch');
ylabel('Cost');

hold on;
for attempt = 1:100
    start = tic;
    [x, y, historyBestY] = hc12(@rosen, 10, 5, lims, maxGen, 0.0005, 10);
    finish = toc;

    stats(attempt) = y;
    statsTime(attempt) = finish;

    plot(historyBestY);
end
hold off;

subplot(1, 3, 2);
histogram(stats, 20);
title('Final value distribution');
xlabel('Cost');

subplot(1, 3, 3);
histogram(statsTime, 20);
title('Evaluation time');
xlabel('Time [s]');
   
function [x, y, historyBestY] = hc12(fn, nBitParam, nParam, lims, maxGen, tol, patience)
    % nParam - number of parameters (dimentions)
    % nBitParam - number of bit per parameter
    % lims (dodParam) - range of parameter values, size(nParam, 2) of [max min] * nParam
    %       params
    % maxGen - max generations allowed
    % tol - required minimal change in objective function for `patience` epochs
    %       if less algorithm will terminate
    % patience - wait for change in cost function for N epoch
    
    % Assert patience is possitive
    assert(patience > 0);
    
    % Termination logic
    terminate = @(dx) abs(dx) < tol;
    
    % Length of logical vector representation
    instSize = nBitParam*nParam;
    
    koef = (2^nBitParam-1) / (lims(2) - lims(1));
    fit = @(val) val / koef + lims(1);

    % Inti transformation matrixes
    M0 = zeros(1, instSize);
    M1 = eye(instSize);
    M2 = genHammDist2(instSize);
    M = [M0; M1; M2];

    K = initK(nParam, nBitParam);
    
    bestH = inf;
    bestHOld = inf;
    bestIndividual = nan;

    historyBestY = nan(maxGen, 1);
    
    count = patience;
    for epoch=1:maxGen
        % Generate surrounding using distance matrix (flip bits where mask is 1
        % to get numbers that has 0, 1, 2 h distance from given K)
        B = repmat(K, size(M, 1), 1);
        B(M == 1) = ~B(M == 1);
    
        % Calculate cost function
        h = calculateH(B, fn, fit, nParam, nBitParam);

        % Find best individual
        [tmpBestVal, tmpBestPos] = min(h);
        if tmpBestVal <= bestH
            bestH = tmpBestVal;
            bestIndividual = B(tmpBestPos, :);
        end
        historyBestY(epoch) = bestH;

        % Think abount terminating if: 1) best not changing, 2) change in best is
        % low (less then tolerance).
        if terminate(bestHOld - bestH) || epoch == maxGen
            count = count - 1;
            if ~count || epoch == maxGen
                % fprintf('Done. Epoch: %i', epoch);
                
                [x, y] = calculateSignleH(bestIndividual, fn, fit, nParam, nBitParam);
                historyBestY = historyBestY(~(isnan(historyBestY)));

                return 
            end
        else
            % Big change detected - reset patience
            count = patience;
        end
        bestHOld = bestH;
        K = B(tmpBestPos, :);
    end

    function h = calculateH(B, fn, fit, nParam, nBitParam)
        h = zeros(size(B, 1), 1);
        for i = 1:size(B, 1)
            k = zeros(nParam, 1); 
            for ii=0:nParam-1
                k(ii+1) = fit(bi2de(grayToBin(B(i, ii*nBitParam+1:(ii+1)*nBitParam))));
            end
            h(i) = fn(k);  
        end
    end

    function [x, y] = calculateSignleH(ind, fn, fit, nParam, nBitParam)
        x = zeros(nParam, 1); 
        for i=0:nParam-1
            x(i+1) = fit(bi2de(grayToBin(ind(i*nBitParam+1:(i+1)*nBitParam))));
        end
        y = fn(x);
    end

    function k = initK(nParam, nBitParam)
        k = zeros(1, nParam*nBitParam);
        for i=0:nParam-1
            k(i*nBitParam+1:(i+1)*nBitParam) = double(dec2bin(randi(2^nBitParam-1), nBitParam) == '1');
        end
    end

    function M = genHammDist2(n)
        % Generate M matrix with all bin numbers with H dist = 2 from zero
        comb = nchoosek(1:n, 2);
        M = zeros(size(comb, 1), n);
        for pos = 1:size(comb, 1)
            M(pos, comb(pos, 1)) = 1;
            M(pos, comb(pos, 2)) = 1;
        end
    end
    
    function bin = grayToBin(gray)
        % Convert Gray code to binary
        bin = zeros(size(gray));
        bin(1) = gray(1);
        for i = 2:length(gray)
            bin(i) = xor(bin(i - 1), gray(i));
        end
    end
end



