clc; clf; clear;

lims = [-5 10];
maxGen = 1000;

% [x, y, historyBestY] = ga(@rosen, 100, 0, 1, 1, 10, 4, lims, maxGen, 20, 0.0005, maxGen);
% 
% figure(1);
% plot(historyBestY);

stats = zeros(100, 1);
statsTime = zeros(100, 1);

f = figure(1);

ax = subplot(1, 3, 1);
% ax.YLim = [0 20];
title('Cost progress');
xlabel('Epoch');
ylabel('Cost');

hold on;
for attempt = 1:100
    start = tic;
    [x, y, historyBestY] = ga(@rosen, 100, 0.1, 0.5, 0.5, 10, 5, lims, maxGen, 10, 0.05, maxGen);
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


function [x, y, historyBestY] = ga(fn, nP, pS, pC, pM, nBitParam, nParam, lims, maxGen, tSize, tol, patience)
    % nP - population size
    % pS - ratio of elite selection, 0 <= pS <= 1
    % pC - probability of crossover, 0 <= pC <= 1
    % pM - probability of mutation, 0 <= pM <= 1
    % nParam - number of parameters (dimentions)
    % nBitParam - number of bit per parameter
    % lims (dodParam) - range of parameter values, size(nParam, 2) of [max min] * nParam
    %       params
    % maxGen - max generations allowed
    % tSize - size of tournament
    % tol - required minimal change in objective function for `patience` epochs
    %       if less algorithm will terminate
    % patience - wait for change in cost function for N epoch
    
    % Assert patience is possitive
    assert(patience > 0);
    
    % Termination logic
    terminate = @(dx) abs(dx) < tol;
    
    % Length of logical vector representation
    instSize = nBitParam*nParam;
    
    vMax = 2^nBitParam-1;
    k = vMax / (lims(2) - lims(1));
    fit = @(val) val / k + lims(1);

    % Number of elite
    eliteN = floor(nP * pS);
    if mod(eliteN, 2)
        eliteN = eliteN + 1;
    end

    % Get random population
    pop = initPop(nP, nParam, nBitParam, instSize, vMax);

    count = patience;
    
    bestH = inf;
    bestHOld = inf;
    bestIndividual = nan;

    historyBestY = nan(maxGen, 1);
    for epoch=1:maxGen
        % Calculate cost function
        h = calculateH(pop, fn, fit, nParam, nBitParam);

        % Find best individual
        [tmpBestVal, tmpBestPos] = min(h);
        if tmpBestVal <= bestH
            bestH = tmpBestVal;
            bestIndividual = pop(tmpBestPos, :);
        end
        historyBestY(epoch) = bestH;

        % Think abount terminating if: 1) best not changing, 2) change in best is
        % low (less then tolerance).
        if terminate(bestHOld - bestH) || epoch == maxGen
            count = count - 1;
            if ~count || epoch == maxGen
                % fprintf('Done. Epoch: %i', epoch);
                
                [x, y] = calculateSingleH(bestIndividual, nParam, nBitParam, fn, fit);

                historyBestY = historyBestY(~(isnan(historyBestY)));
                
                return 
            end
        else
            % Big change detected - reset patience
            count = patience;
        end
        bestHOld = bestH;

        % Main population array
        popNew = zeros(nP, instSize);

        % Sort ascending by h
        [~, sortIdx] = sort(h,'ascend');

        popEliteIdx = sortIdx(1:eliteN);
        popOtherIdx = sortIdx(eliteN+1:end);

        % Resolve elite
        popNew(1:eliteN, :) = pop(popEliteIdx, :);
        % Drop elite from current pop to perform selection and mutitation
        pop = pop(popOtherIdx, :);
        h = h(popOtherIdx);

        % Resolve evolution
        for iChunk=0:2:((nP-eliteN)/2)-1
            
            chunk = doSelect(pop, h, tSize, instSize);
            % Resolve crossover
            if rand() <= pC
                chunk = doCross(chunk, nParam, nBitParam, false);
            end
            % Perform mutation on both individuals
            for iInd = 1:2
                % Resolve mutation should occur
                if rand() <= pM
                    newInd = doMutation(chunk(iInd, :));
                    chunk(iInd, :) = newInd;
                end
            end
            popNew(iChunk*2+1:iChunk*2+2, :) = chunk;
        end
        pop = popNew;
    end

    % Tournament selection (ensure elitism before passing popuation)
    function selection = doSelect(pop, h, tSize, instSize)
        % pop - matrix of individuals (population), size = (nP, instSize)
        % h - matrix of cost, size = (nP, 1)
        % tSize - tournament size, int
        % instSize - instance size, int
        % Returns (2, instSize) selected instances
        pop = pop(:,:);
        popSize = size(pop);

        selection = zeros(2, instSize);
        for i = 1:2
            % Select sub population for tournament by index
            ind = 1:popSize(1);
            ind = randsample(ind, tSize);

            tPop = pop(ind, :);
            tH = h(ind);

            % Evaluate objective and select best
            [~, p] = min(tH);
            selection(i, :) = tPop(p, :);
        end
    end

    % 1 point crossover
    function cross = doCross(pop, nParam, nBitParam, cutRnd)
        % pop - sub population of 2 individuals to cross
        pop = pop(:,:);
        popSize = size(pop);


        % Select cut point
        if cutRnd
            cut = 1 + randi(popSize(2) - 2);
        else
            cut = randi(nParam - 1) * nBitParam;
        end 

        % Swap parts
        tmp = pop(1, 1:cut);
        pop(1, 1:cut) = pop(2, 1:cut);
        pop(2, 1:cut) = tmp;

        cross = pop;
    end

    %  1 point mutation
    function mutation = doMutation(inst)
        % inst - original instance

        inst = inst(:);

        % Select index to apply mutation
        mutI = randi(length(inst));
        % Make sure to use copies
        % Flip value (aka mutate)
        inst(mutI) = ~inst(mutI);
        mutation = inst;
    end

    function pop = initPop(nP, nParam, nBitParam, instSize, vMax)
        % Main population array
        pop = zeros(nP, instSize);
        % Initial values
        for i=1:nP  % Each instance
            for ii=0:nParam-1  % Each parameter
                % Generate random initial value from range of 0 to
                % bin2dec('111...nBitParam')
                val = double(dec2bin(randi(vMax), nBitParam) == '1');
                pop(i, ii*nBitParam+1:(ii+1)*nBitParam) = val;
            end
        end
    end

    function h = calculateH(pop, fn, fit, nParam, nBitParam)
        h = zeros(size(pop, 1), 1);
        for i=1:size(pop, 1)
            p = zeros(nParam, 1);
            for ii=0:nParam-1
                p(ii+1) = fit(bi2de(pop(i, ii*nBitParam+1:(ii+1)*nBitParam)));
            end
            h(i) = fn(p);
        end
    end

    function [x, y] = calculateSingleH(ind, nParam, nBitParam, fn, fit)
        x = zeros(nParam, 1);
        for i=0:nParam-1
            x(i+1) = fit(bi2de(ind(i*nBitParam+1:(i+1)*nBitParam)));
        end
        y = fn(x);
    end

end

