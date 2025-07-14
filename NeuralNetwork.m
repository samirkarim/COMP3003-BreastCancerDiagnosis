%%%%%%%%%%  Training a neural network on breast cancer data %%%%%%%%%%%

%Step 1: Load the dataset from the CSV file

datasetFile = 'data.csv';

dataTable = readtable(datasetFile);

%Step 2: Convert tables to matrices and y values from B and M to 0 and 1
%respectively

numericalData = dataTable(:, 3:end);

xMatrix = table2array(numericalData);

y = dataTable.Var2;
yNumerical = categorical(y);
yNumerical = grp2idx(yNumerical) - 1;

%Step 3: Set up the cross-validation

kValues = [3, 5, 7, 10, 15, 20, floor(sqrt(size(xMatrix, 1)))];

accuracyResults = zeros(length(kValues), 1);

%Step 4: Enter the cross-validation loop

for kIndex = 1:length(kValues)

    k = kValues(kIndex);

    cv = cvpartition(size(xMatrix, 1), 'KFold', k);

    foldAccuracies = zeros(k, 1);

    %Step 5: Iterate through the folds

    for fold = 1:k
        
        trainIndex = training(cv, fold);
        testIndex = test(cv, fold);

        %Step 6: Splitting the data into testing and training
        
        xTrainFold = xMatrix(trainIndex, :);
        yTrainFold = yNumerical(trainIndex);
        xTestFold = xMatrix(testIndex, :);
        yTestFold = yNumerical(testIndex);

        %Step 7: Normalise the data (wasn't done earlier due to issues..)
        
        meanValues = mean(xTrainFold);
        stdValues = std(xTrainFold);

        % Z-score xTrain and xTest folds
        xTrainFold = (xTrainFold - meanValues) ./ stdValues;
        xTestFold = (xTestFold - meanValues) ./ stdValues;

        %Step 8: Train the patternnet model
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);

        %Training parameters
        net.trainParam.epochs = 100;
        net.trainParam.lr = 0.01;
        net.trainParam.goal = 1e-5;
        net.trainParam.max_fail = 10;

        %Train the network
        net = train(net, xTrainFold', yTrainFold');
        
        %Predict the test set
        yPredicted = net(xTestFold');

        %Round to convert probabilities to binary answer
        yPredictedBinary = round(yPredicted);

        %Evaluate the accuracy
        foldAccuracies(fold) = sum(yPredictedBinary == yTestFold') / numel(yTestFold);
    end
    
    %display results for current k
    accuracyResults(kIndex) = mean(foldAccuracies);
    fprintf('k = %d, Average Accuracy: %f\n', k, accuracyResults(kIndex));
end

[bestAccuracy, bestKIndex] = max(accuracyResults);
bestK = kValues(bestKIndex);

fprintf('\nBest k: %d (Accuracy: %f)\n', bestK, bestAccuracy);

%Step 9: Plot the results

figure;
plot(kValues, accuracyResults, '-o');
xlabel('Number of K folds');
ylabel('Average accuracy');
title('Cross Validation Accuracy vs. Number of folds using a NN');
