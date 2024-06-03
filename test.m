% Load the trained network
load('mtrainednet.mat', 'net');

dataDir = 'C:\classification\HAM10000-aug';
imgSize = 64; % Image size used during training

% Create an image datastore
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Resize images
imds.ReadFcn = @(filename)imresize(imread(filename), [imgSize imgSize]);

% Split data into training and testing sets
[~, imdsTest] = splitEachLabel(imds, 0.8, 'randomize');

% Classify test images
YPred = classify(net, imdsTest);
YTest = imdsTest.Labels;

% Calculate accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Test accuracy: ', num2str(accuracy)]);

% Display the confusion matrix
confMat = confusionmat(YTest, YPred);
confMat = confMat ./ sum(confMat, 2); % Normalize the confusion matrix
figure;
heatmap(categories(YTest), categories(YTest), confMat);
title('Confusion Matrix');
xlabel('Predicted');
ylabel('Actual');

% Visualize some test images with their predicted labels
numImages = 10; % Number of images to display
idx = randperm(numel(YTest), numImages); % Randomly select images
figure;
for i = 1:numImages
    subplot(2, 5, i);
    img = readimage(imdsTest, idx(i));
    imshow(img);
    title(sprintf('Pred: %s\nActual: %s', string(YPred(idx(i))), string(YTest(idx(i)))));
end

