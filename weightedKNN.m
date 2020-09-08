function [trainedClassifier, validationAccuracy] = weightedKNN(newCombdataformachinelearning)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% Returns a trained classifier and its accuracy.
%
%  Input:
%      trainingData: A table containing the same predictor and response
%       columns as those imported into matlab.
%
%  Output:
%      trainedClassifier: A struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: A function to make predictions on new
%       data.
%
%      validationAccuracy: A double containing the accuracy in percent.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument trainingData.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training. 

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = newCombdataformachinelearning;
predictorNames = {'SVMacc', 'Arctanacc', 'Arccosacc', 'SVMang', 'Arctanang', 'Arccosang', 'SVMorien', 'Arctanorien', 'Arccosorien', 'SVMmag', 'Arctanmag', 'Arccosmag'};
predictors = inputTable(:, predictorNames);
response = inputTable.Activities;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true, ...
    'ClassNames', categorical({'Climbing Down stairs'; 'Climbing up stairs'; 'Lying on a Flat surface'; 'Sitting'; 'Standing'; 'Walking'}));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'Arccosacc', 'Arccosang', 'Arccosmag', 'Arccosorien', 'Arctanacc', 'Arctanang', 'Arctanmag', 'Arctanorien', 'SVMacc', 'SVMang', 'SVMmag', 'SVMorien'};
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype)');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = newCombdataformachinelearning;
predictorNames = {'SVMacc', 'Arctanacc', 'Arccosacc', 'SVMang', 'Arctanang', 'Arccosang', 'SVMorien', 'Arctanorien', 'Arccosorien', 'SVMmag', 'Arctanmag', 'Arccosmag'};
predictors = inputTable(:, predictorNames);
response = inputTable.Activities;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 5);

% Compute validation predictions
validationAccuracy = 1-kfoldLoss(partitionedModel)
% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

