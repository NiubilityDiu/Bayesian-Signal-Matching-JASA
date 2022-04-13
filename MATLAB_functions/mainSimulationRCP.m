clear
clc

%% test convolution signal function
LocalBool = 1;
switch LocalBool
    case 1
        ParentDir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode';
        Seed = 1;
        SimNum = 1;
    otherwise 
        ParentDir = '/home/mtianwen';
        Seed = str2num(getenv('SLURM_ARRAY_TASK_ID'));
        SimNum = str2num(getenv('SLURM_ARRAY_TASK_ID'));
end
rng(Seed);

% some global constants
TargetKeys = ['T' 'H' 'O' 'M' 'P' 'S' 'O' 'N'];
SeqNumTrn = 10;
% SeqNumTest = 5;
NoiseType = 'normal';
VarNum = 1;
% OrderType = 'spatial';
SimType = lower('init');


SimERTMatDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'SIM_summary' ...
    filesep 'Chapter_2' filesep 'SimTrueValue.mat'];
SimTrueFile = load(SimERTMatDir);
TarERP = SimTrueFile.TargetERP.'; 
NtarERP = SimTrueFile.NonTargetERP.';

% TarERP = [TarERP TarERP];
% NtarERP = [NtarERP NtarERP];
[~, E] = size(TarERP);

StimulusOnTime = 5;
StimulusPauseTime = 1;
WindowLen = 30;


% minimize the contrast between T and NT for now
TarERP = TarERP / 10;
NtarERP = NtarERP / 10;

%% Generate StimCode, StimType based on TargetKey
StimCode = [];
StimType = [];
for KeyID=1:length(TargetKeys)
    [StimTypeKeyID, StimCodeKeyID] = generateStimTypeRCP(TargetKeys(KeyID), SeqNumTrn, Seed);
    StimType = [StimType; StimTypeKeyID];
    StimCode = [StimCode; StimCodeKeyID];
end

%% generate the pseudo dataset in a long format
[SimSignalTrn,StimTypeLongTrn, StimCodeLongTrn] = createSignalConvolRCP(TarERP, NtarERP, ...
    StimType, StimCode, NoiseType, VarNum, StimulusOnTime, StimulusPauseTime, WindowLen);
SimTrnObj = struct();
SimTrnObj.Signal = SimSignalTrn;
SimTrnObj.Type = StimTypeLongTrn;
SimTrnObj.Code = StimCodeLongTrn;
% GroupNum
SimTrnObj.GroupNum = repmat(12, size(StimTypeLongTrn, 1), 1);
% Begin
SimTrnObj.Begin = StimCodeLongTrn > 0;

%% extract long simulation signal
[SimTrnTruncObj] = truncMatTrainRCP(SimTrnObj, WindowLen);

%% confirm useful signals
TarMeanTrnTrunc = reshape(mean(SimTrnTruncObj.Signal(SimTrnTruncObj.Type == 1, :), 1), [WindowLen, E]);
NtarMeanTrnTrunc = reshape(mean(SimTrnTruncObj.Signal(SimTrnTruncObj.Type == -1, :), 1), [WindowLen, E]);
% TarMeanTestTrunc = reshape(mean(SimSignalTestTrunc(SimTypeTestTrunc == 1, :), 1), [WindowLen, E]);
% NtarMeanTestTrunc = reshape(mean(SimSignalTestTrunc(SimTypeTestTrunc == -1, :), 1), [WindowLen, E]);

TimeSeq = 0:1/30:WindowLen/30;
TimeSeq = TimeSeq * 1000;
TimeSeq = TimeSeq(2:end);

h = figure;
plot(TimeSeq, TarMeanTrnTrunc, 'r-', 'LineWidth', 2);
hold on
plot(TimeSeq, NtarMeanTrnTrunc, 'b-', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Amplitude (muV)');
title('Channel 1', 'FontSize', 20);
legend('Train Target', 'Train Non-target', 'FontSize', 15); 
ylim([0, 1.2]);
    

%% save the dataset and plot
folder_dir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'SIM_files' ...
    filesep 'Chapter_2' filesep sprintf('sim_%s', SimType)];
mkdir(folder_dir);
folder_dir = [folder_dir filesep sprintf('sim_%s_%d', SimType, SimNum)];
mkdir(folder_dir);

SimPlotDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'SIM_files' ...
    filesep 'Chapter_2' filesep sprintf('sim_%s', SimType) filesep ...
    sprintf('sim_%s_%d', SimType, SimNum) filesep ...
    sprintf('sim_%s_%d_empirical_mean_function.fig', SimType, SimNum)];
savefig(h, SimPlotDir);

%%
SimERPMatDirTrn = [folder_dir filesep sprintf('sim_%s_%d_train.mat', SimType, SimNum)];

SimObj = SimTrnObj;
SimTruncObj = SimTrnTruncObj;
SeqNum = SeqNumTrn;
save(SimERPMatDirTrn, 'SimObj', 'SimTruncObj', 'VarNum', 'TargetKeys', 'SeqNum');