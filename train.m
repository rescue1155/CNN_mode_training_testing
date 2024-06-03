% Set the random seed for reproducibility
rng(42);

dataDir = 'C:\classification\HAM10000-aug';
imgSize = 64; % Commonly used image size for CNNs

% Create an image datastore
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Resize images
imds.ReadFcn = @(filename)imresize(imread(filename), [imgSize imgSize]);

% Split data into training and testing sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomize');

% Define the network architecture
layers = [
    imageInputLayer([imgSize imgSize 3])
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(256)
    dropoutLayer(0.5)
    fullyConnectedLayer(numel(categories(imds.Labels)))
    softmaxLayer
    classificationLayer];

% Set training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(imdsTrain, layers, options);

% Classify test images
YPred = classify(net, imdsTest);
YTest = imdsTest.Labels;

% Calculate accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Test accuracy: ', num2str(accuracy)]);

% Save the trained network
save('skin_cancer_classification_model.mat', 'net');
