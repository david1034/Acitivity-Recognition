function [trainedClassifier, validationAccuracy] = gaussianNaiveBayes(newCombdataformachinelearning)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% Returns a trained classifier and its accuracy. 

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

% Expand the Distribution Names per predictor
% Numerical predictors are assigned either Gaussian or Kernel distribution and categorical predictors are assigned mvmn distribution
% Gaussian is replaced with Normal when passing to the fitcnb function
distributionNames =  repmat({'Normal'}, 1, length(isCategoricalPredictor));
distributionNames(isCategoricalPredictor) = {'mvmn'};

if any(strcmp(distributionNames,'Kernel'))
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'Kernel', 'Normal', ...
        'Support', 'Unbounded', ...
        'DistributionNames', distributionNames, ...
        'ClassNames', categorical({'Climbing Down stairs'; 'Climbing up stairs'; 'Lying on a Flat surface'; 'Sitting'; 'Standing'; 'Walking'}));
else
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'DistributionNames', distributionNames, ...
        'ClassNames', categorical({'Climbing Down stairs'; 'Climbing up stairs'; 'Lying on a Flat surface'; 'Sitting'; 'Standing'; 'Walking'}));
end

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
trainedClassifier.predictFcn = @(x) naiveBayesPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'Arccosacc', 'Arccosang', 'Arccosmag', 'Arccosorien', 'Arctanacc', 'Arctanang', 'Arctanmag', 'Arctanorien', 'SVMacc', 'SVMang', 'SVMmag', 'SVMorien'};
trainedClassifier.ClassificationNaiveBayes = classificationNaiveBayes;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = newCombdataformachinelearning;
predictorNames = {'SVMacc', 'Arctanacc', 'Arccosacc', 'SVMang', 'Arctanang', 'Arccosang', 'SVMorien', 'Arctanorien', 'Arccosorien', 'SVMmag', 'Arctanmag', 'Arccosmag'};
predictors = inputTable(:, predictorNames);
response = inputTable.Activities;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationNaiveBayes, 'KFold', 5);

% Compute validation predictions
validationAccuracy = 1-kfoldLoss(partitionedModel)

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
