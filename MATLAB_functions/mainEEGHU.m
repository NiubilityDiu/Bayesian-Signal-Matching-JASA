%% this is the file to fit swlda to the training dataset and predict on the training set itself.
clear
clc

%% extract signals and parameters
ParentDir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode/EEG_MATLAB_data/TRN_files/';

BandUpp = 15;
DecFactor = 6;
ProtocolName = 'U';
SubjectID = '352';
FileID = '001';
Mode = 'DRYAAC';
DataType = 'TRN';
EEGDownObj = preProcessSignal(ProtocolName, SubjectID, FileID, Mode, DataType, BandUpp, DecFactor, ParentDir);
ChannelName = EEGDownObj.ChannelName;
[~, E] = size(EEGDownObj.Signal);
StdWinLen = EEGDownObj.Fs / DecFactor;
WindowLen = StdWinLen; % usually * 0.8
[EEGTruncObj] = truncMatTrain(EEGDownObj, WindowLen);

%% perform swLDA analysis
MaxSelect = round(0.3 * WindowLen * E);
% train-test split scheme (problematic, because the keyboard will change)
% do not consider split right now
[B, SE, Pval, InModel, Stats] = trainSWLDAmatlab(EEGTruncObj.Signal, EEGTruncObj.Type, MaxSelect);

disp(sum(InModel));
disp(sum(reshape(InModel.', [], E).',2));
% compute the classifier score
YScoreTrn = EEGTruncObj.Signal(:, InModel) * B(InModel,:);
EEGTruncObj.Score = YScoreTrn;

%% histogram and density plot
Hist = produceHistDensityPlot(YScoreTrn, EEGTruncObj.Type, [], [], ProtocolName, SubjectID);
HistDir = [ParentDir filesep sprintf('%s%s', ProtocolName, SubjectID) filesep ...
    sprintf('%s%s_%s_%s_%s_Score_Histogram_Density.fig', ProtocolName, SubjectID, FileID, Mode, DataType)];
savefig(Hist, HistDir);

%% determine target key first
TotalNum = size(YScoreTrn, 1);
TargetNum = size(EEGTruncObj.Type == 1, 1);
TotalTargetKeyNum = 20;
SeqNum = 15;
TagsTarget = EEGTruncObj.Tags(EEGTruncObj.Type == 1,:);
TargetKeyUnique = zeros(TotalTargetKeyNum, 1);
for i=1:TotalTargetKeyNum
   TagTargeti = reshape(TagsTarget((i-1)*SeqNum*2+1:i*SeqNum*2, :), [], 1); 
   TagTargeti = TagTargeti(TagTargeti > 0, :);
   TargetKeyUnique(i,:) = mode(TagTargeti, 'All');
end
TargetKeyNums = reshape(repmat(TargetKeyUnique, [1 SeqNum*2]).', [], 1);

%% frequentist method to estimate mu1, mu0, and sigma^2.
MuTar = mean(YScoreTrn(EEGTruncObj.Type == 1, :));
MuNtar = mean(YScoreTrn(EEGTruncObj.Type ~=1, :));
SigmaAll = sqrt(var(YScoreTrn));

% save the swlda output of training set
SwldaDir = [ParentDir filesep sprintf('%s%s', ProtocolName, SubjectID) filesep ...
    sprintf('%s%s_%s_%s_%s_swLDA_Output.mat', ProtocolName, SubjectID, FileID, Mode, DataType)];
save(SwldaDir, 'B', 'SE',  'Pval', 'InModel', 'E', 'DecFactor', 'WindowLen', ...
    'MuTar', 'MuNtar', 'SigmaAll');

%% prediction accuracy
GridSize = [7 12];
TotalKeyNum = prod(GridSize);

SwldaObj = struct();
SwldaObj.B = B;
SwldaObj.InModel = InModel;
SwldaObj.MuTar = MuTar;
SwldaObj.MuNtar = MuNtar;
SwldaObj.SigmaAll = SigmaAll;

% we already compute YScoreTrn, no need to use SwldaObj again.
[KeyProbArr, EntropyArr, KeySelectArr, ~] = swLDAPredict(SwldaObj, EEGTruncObj.Signal, ...
EEGTruncObj.GroupNum, EEGTruncObj.Tags, YScoreTrn, GridSize, TotalTargetKeyNum, SeqNum);

%% save the prediction accuracy results
KeySelectArr = reshape(KeySelectArr, [SeqNum+1, TotalTargetKeyNum]);
KeySelectArr = KeySelectArr(2:end,:);
KeyProbArr = reshape(KeyProbArr.', [TotalKeyNum, SeqNum+1, TotalTargetKeyNum]);
KeyProbArr = KeyProbArr(:,2:end,:);
EntropyArr = reshape(EntropyArr, [SeqNum+1, TotalTargetKeyNum]);
EntropyArr = EntropyArr(2:end, :);

PredictDir = [ParentDir filesep sprintf('%s%s', ProtocolName, SubjectID) filesep ...
    sprintf('%s%s_%s_%s_%s_Prediction.mat', ProtocolName, SubjectID, FileID, Mode, DataType)];
save(PredictDir, 'KeySelectArr', 'KeyProbArr',  'EntropyArr', 'TargetKeyUnique');

%% produce the prob trend w.r.t sequence replications for each target char
% FigureProb = figure;
% for TargetID=1:TotalTargetKeyNum
%     plot(1:SeqNum, KeyProbArr(TargetKeyUnique(TargetID,:),:,TargetID), 'LineWidth', 2);
%     hold on
% end

FigureProbSubplot = figure;
for TargetID=1:TotalTargetKeyNum
    subplot(5,4,TargetID)
    plot(1:SeqNum, KeyProbArr(TargetKeyUnique(TargetID,:),:,TargetID), 'LineWidth', 2);
    title(sprintf('ID%d, CharID%d', TargetID, TargetKeyUnique(TargetID,:)), 'FontSize', 15);
end
sgtitle('Target Character Probability Trend', 'FontSize', 20);

FigureProbSubplotDir = [ParentDir filesep sprintf('%s%s', ProtocolName, SubjectID) filesep ...
    sprintf('%s%s_%s_%s_%s_Target_Prob_Trend.fig', ProtocolName, SubjectID, FileID, Mode, DataType)];
savefig(FigureProbSubplot, FigureProbSubplotDir);
