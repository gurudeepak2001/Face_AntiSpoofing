% Model training 
clear all
close all
clc
%% Input for Training set 
rootFolder = fullfile('G:\combine project\frame skipping\O+C+I\TrainSet');
categories  = {'real','attack'};
trainingset1 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Input for development set 
rootFolder = fullfile('G:\combine project\frame skipping\O+C+I\TestSet');
categories  = {'real','attack'};
developmentset1 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Input for Testing set 
rootFolder = fullfile('G:\combine project\frame skipping\MSU dataset\testing set');
categories = {'real','attack'};
testingsetdata1 = imageDatastore(fullfile(rootFolder, categories),  'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Extracting  labels for training, development, and test set
trainingLabels1 = trainingset1.Labels;
developmentlabel1 = developmentset1.Labels;
testinglabel1 = testingsetdata1.Labels;
%% Import Deep learning model
net = densenet201;
%% Features extraction based on the last average pooling layer for the training set 
featureLayer =   'avg_pool';
% Resize image
trainingset1.ReadFcn = @(filename)readAndPreprocessImage(filename);
trainingFeatures1 = activations(net, trainingset1, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer for the development set 
% Resize image
 developmentset1.ReadFcn = @(filename)readAndPreprocessImage(filename);
% Extract development set features using the CNN
developmentFeatures1 = activations(net,developmentset1, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer for the testing set 
% Extract testing set features using the CNN
testingsetdata1.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures1 = activations(net, testingsetdata1, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%%
% Converting data for BiLSTM training 
rng(13);
trainf = {};
trainf{end+1} =  trainingFeatures1;

trainlabl = {};
trainlabl{end+1} = trainingLabels1';

train1 = {};
train1{end+1} = developmentFeatures1;
% 
train2 = {};
train2{end+1} = developmentlabel1';


% Training for BiLSTM Model 
numFeatures = 1920;
 numHiddenUnits =500;
numClasses = 2;
layers1 = [ ...
    sequenceInputLayer(numFeatures)
         bilstmLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options1 = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

lstm1 = trainNetwork(trainf',trainlabl,layers1,options1);

[~, devlp_scores1] = classify(lstm1, developmentFeatures1);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel1);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores1 =devlp_scores1';
% choosing threshold
 [~,~,Info]=vl_roc(numericLabels1,devlpscores1(:,1));
 EER = Info.eer*100;
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [~, test_scores1] = classify(lstm1, testingFeatures1);
 testscores1 = test_scores1';
 numericLabels = grp2idx(testinglabel1);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores1(numericLabels==1);
 attack_scores2 =  testscores1(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER1 = (FAR+FRR)/2
 
[x1,y1,~,AUC1] = perfcurve(numericLabels, testscores1(:,1),1);
AUC1
plot(x1,y1,'-g','LineWidth',1.8,'MarkerSize',1.8)
grid on
hold on
 
 %%
% Converting data for GRU training 
rng(13);
trainf = {};
trainf{end+1} =  trainingFeatures1;

trainlabl = {};
trainlabl{end+1} = trainingLabels1';

train1 = {};
train1{end+1} = developmentFeatures1;
% 
train2 = {};
train2{end+1} = developmentlabel1';


% Training for GRU Model 
numFeatures = 1920;
 numHiddenUnits =20;
numClasses = 2;
layers2 = [ ...
    sequenceInputLayer(numFeatures)
         gruLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options2 = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

lstm2 = trainNetwork(trainf',trainlabl,layers2,options2);

[~, devlp_scores2] = classify(lstm2, developmentFeatures1);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel1);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores2 =devlp_scores2';

 [~,~,Info]=vl_roc(numericLabels1,devlpscores2(:,1));
 EER = Info.eer*100;
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [~, test_scores2] = classify(lstm2, testingFeatures1);
 testscores2 = test_scores2';
 numericLabels = grp2idx(testinglabel1);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores2(numericLabels==1);
 attack_scores2 =  testscores2(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER2 = (FAR+FRR)/2
 
[x2,y2,~,AUC2] = perfcurve( numericLabels, testscores2(:,1),1);
AUC2
 plot(x2,y2,'-m','LineWidth',1.8,'MarkerSize',1.8)
 grid on
 hold on
 
%%
% Converting data for LSTM training 

rng(1);
trainf = {};
trainf{end+1} =  trainingFeatures1;

trainlabl = {};
trainlabl{end+1} = trainingLabels1';

train1 = {};
train1{end+1} = developmentFeatures1;
% 
train2 = {};
train2{end+1} = developmentlabel1';

% Training for LSTM Model 

numFeatures = 1920;
 numHiddenUnits =1000;
numClasses = 2;
layers3 = [ ...
    sequenceInputLayer(numFeatures)
         lstmLayer(numHiddenUnits,'OutputMode','sequence')
     fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options3 = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

lstm3 = trainNetwork(trainf',trainlabl,layers3,options3);

[~, devlp_scores3] = classify(lstm3, developmentFeatures1);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel1);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores3 =devlp_scores3';
 [~,~,Info]=vl_roc(numericLabels1,devlpscores3(:,1));
 EER = Info.eer*100;
 threashold = Info.eerThreshold;
 % Evaluation for testing set for HTER 
 [~, test_scores3] = classify(lstm3, testingFeatures1);
 testscores3 = test_scores3';
 numericLabels = grp2idx(testinglabel1);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores3(numericLabels==1);
 attack_scores2 =  testscores3(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER3 = (FAR+FRR)/2
 
[x3,y3,threshold,AUC3] = perfcurve(numericLabels, testscores3(:,1),1);
AUC3
 plot(x3,y3,'-b','LineWidth',1.8,'MarkerSize',1.8)
 grid on
 hold on
 
%%
% Ensemble Learning
finaldevlopscores =  devlpscores1 +   devlpscores2 +  devlpscores3;
finaltestscores =  testscores1  + testscores2  + testscores3 ;

% Final Meta-model (BiLSTM) TRAINING 
rng(13) % For reproducibility
trainf = {};
trainf{end+1} =  finaldevlopscores';

trainlabl = {};
trainlabl{end+1} = developmentlabel1';

numFeatures =2;
 numHiddenUnits =20;
numClasses = 2;
layers4 = [ ...
    sequenceInputLayer(numFeatures)
   lstmLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

maxEpochs = 100;
miniBatchSize = 32;

options5 = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',maxEpochs, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

finallstm = trainNetwork(trainf',trainlabl,layers4,options5);

[predictedLabels4, devlp_scoresfinal] = classify(finallstm, finaldevlopscores');
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel1);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscoresfianl =devlp_scoresfinal';

 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscoresfianl(:,1));
 EER = Info.eer*100;
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [~, test_scoresfinal] = classify(finallstm, finaltestscores');
 testscoresfinal = test_scoresfinal';
 numericLabels = grp2idx(testinglabel1);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 = testscoresfinal(numericLabels==1);
 attack_scores2 = testscoresfinal(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 finalhterBilstm = (FAR+FRR)/2
[x4,y4,threshold,AUC4] = perfcurve(numericLabels, testscoresfinal(:,1),1);
AUC4
plot(x4,y4,'-c','LineWidth',1.8,'MarkerSize',1.8)
grid on
hold off
legend('O&M&I to C (Model 1)','O&M&I to C  (Model 2)','O&M&I to C  (Model 3)','O&M&I to C  (Meta-Model)')

  %%
 % Resize images 
     function Iout = readAndPreprocessImage(filename)

       I = imread(filename);

         if ismatrix(I)
            I = cat(3,I,I,I);
         end
     

           Iout = imresize(I, [224 224]);
            
     end
    %%
 % Function for early stopping
 
 function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end