clear
clc

%% test convolution signal function
LocalBool = 1;
switch LocalBool
    case 1
        ParentDir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode';
        % ParentDir = 'K:/Dissertation/Dataset and Rcode';
        Seed = 1;
        % SimNum = 1;
    otherwise 
        ParentDir = '/home/mtianwen';
        Seed = str2num(getenv('SLURM_ARRAY_TASK_ID'));
        % SimNum = str2num(getenv('SLURM_ARRAY_TASK_ID'));
end
rng(Seed);

%% Global information for subject
SubjectNameShort = 'K115';
RawFileName = '001_BCI_TRN';
DataType = 'TRN_files';
BandLow = 0.5;
BandUpp = 10;
DecFactor = 8;
EEGWindowLen = 30;

%% Bandpass filtering and down-sampling 
EEGDownLong = extractRawEEGRCP(SubjectNameShort, RawFileName, ...
    DataType, BandLow, BandUpp, DecFactor, LocalBool);

%% Feature extraction
ReshapeOption = '3';
EEGDownTrunc = truncMatTrainRCP(EEGDownLong, EEGWindowLen, ReshapeOption);

%% Verify the P300 signals
for i=1:16
    plot(mean(EEGDownTrunc.Signal(EEGDownTrunc.Type == 1, :, i)), 'Color', 'red'); 
    hold on
    plot(mean(EEGDownTrunc.Signal(EEGDownTrunc.Type == -1, :, i)), 'Color', 'blue');
    legend('Target', 'Non-target');
end
%% Convert it to covariance-based data object
EEGDownCov = constructCovMatRCP(EEGDownTrunc, [], []);
CovFeatureLen = size(EEGDownCov.Signal, 2);
[b, se, pval, inmodel, stats] = trainSWLDAmatlab(EEGDownCov.Signal, EEGDownCov.Type, ceil(0.5 * CovFeatureLen));
