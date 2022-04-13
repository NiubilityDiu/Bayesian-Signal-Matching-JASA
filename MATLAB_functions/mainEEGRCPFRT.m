%% this is the file to fit swlda to the training dataset and predict on the training set itself.
clear
clc

%% extract signals and parameters
LocalUse = 0;
if LocalUse
    ParentDir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode/EEG_MATLAB_data/TRN_files/';
else
    ParentDir = '/home/mtianwen/EEG_MATLAB_data/TRN_files';
end

BandUpp = 15;
DecFactor = 8;
SubName = 'K106';
FRTName = '001_BCI_TRN';
DataType = 'TRN_files';
EEGDownObj = extractRawEEGRCP(SubName, FRTName, DataType, BandUpp, DecFactor, LocalUse);
ChannelName = EEGDownObj.ChannelName;
[~, E] = size(EEGDownObj.Signal);
StdWinLen = EEGDownObj.Fs / DecFactor;

%% for K177_003_BCI_FRT, we need separate truncation of data.
if strcmp(SubName, "K177") && strcmp(FRTName, "003_BCI_FRT")
    UppBound = 7745;
    EEGDownObj.Signal = EEGDownObj.Signal(1:UppBound,:);
    EEGDownObj.Type = EEGDownObj.Type(1:UppBound,:);
    EEGDownObj.Begin = EEGDownObj.Begin(1:UppBound,:);
    EEGDownObj.PIS = EEGDownObj.PIS(1:UppBound,:);
    EEGDownObj.Code = EEGDownObj.Code(1:UppBound,:);
    EEGDownObj.Text = EEGDownObj.Text(1:25,:);
    EEGDownObj.ID = SubName;
end
%% WindowLen = StdWinLen; % usually * 0.8
WindowLen = 25;
[EEGTruncObj] = truncMatTrainRCP(EEGDownObj, WindowLen);
%% For K177_003_BCI_FRT only
% EEGTruncObj.ID = SubName;
%%
TargetMean = mean(EEGTruncObj.Signal(EEGTruncObj.Type(:,1)==1,:), 1);
NtargetMean = mean(EEGTruncObj.Signal(EEGTruncObj.Type(:,1)==-1,:), 1);
ChannelDim = size(TargetMean, 2) / WindowLen;

FigureTest = figure;
for e_id=1:ChannelDim
    subplot(4, 4, e_id);
    plot(1:WindowLen, TargetMean(:, (e_id-1)*WindowLen+1:e_id*WindowLen), 'r');
    hold on
    plot(1:WindowLen, NtargetMean(:, (e_id-1)*WindowLen+1:e_id*WindowLen), 'b');
    hold off
    title_id = "channel" + ' ' + string(e_id);
    title(title_id);
end
sgtitle(SubName);
savefig(FigureTest, [ParentDir filesep 'K106' filesep 'K106_sample_MATLAB_direct.fig'])