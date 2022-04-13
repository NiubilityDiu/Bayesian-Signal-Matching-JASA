clear
clc

%% setup the directory
LocalBool = 0;
switch LocalBool
    case 1
        ParentDir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode';
        MatlabFunDir = [ParentDir filesep 'Chapter_3' filesep 'MATLAB_functions'];
        cd(MatlabFunDir); 
    otherwise 
        ParentDir = '/home/mtianwen';
end

N = 4;
K = 3;
ClusterName = sprintf('N_%d_K_%d', N, K);
OptionID = 7;
SigmaVal = 5.0;
RhoVal = 0.5;
% IterNum = 0;
% IterNum = str2num(getenv('SLURM_ARRAY_TASK_ID'));
RCPUnitFlashNum = 12;
SignalLength = 25;
SeqSizeTrain = 10;

for IterNum=0:99
ScenarioName = sprintf('%s_K114_based_option_%d_sigma_%.1f_rho_%.1f', ClusterName, ...
    OptionID, SigmaVal, RhoVal);
SimDataFolderDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'SIM_files' ...
    filesep 'Chapter_3' filesep 'numpyro_output' filesep ClusterName ...
    filesep ScenarioName filesep sprintf('iter_%d', IterNum)];

% create swLDA folder
SimDataswLDAFolderDir = [SimDataFolderDir filesep 'swLDA'];
mkdir(SimDataswLDAFolderDir);

%% import the training (source data)
SimDataDir = [SimDataFolderDir filesep 'sim_dat.json'];
SimDataFile = fopen(SimDataDir); % open the file
SimDataRaw = fread(SimDataFile, inf); % read the contents
SimDataStr = char(SimDataRaw'); % transform
fclose(SimDataFile); % close the file
SimData = jsondecode(jsondecode(jsonencode(SimDataStr)));

%% import the test dataset.
SimTestDataDir = [SimDataFolderDir filesep 'sim_dat_test.json'];
SimTestDataFile = fopen(SimTestDataDir); % open the file
SimTestDataRaw = fread(SimTestDataFile, inf); % read the contents
SimTestDataStr = char(SimTestDataRaw'); % transform
fclose(SimTestDataFile); % close the file
SimTestData = jsondecode(jsondecode(jsonencode(SimTestDataStr)));

%% import mcmc samples with specified sequence size
for SeqSize=1:10
    fprintf('Seq Size: %d\n', SeqSize);
    MCMCDataDir = [SimDataFolderDir filesep sprintf('mcmc_seq_size_%d.mat', SeqSize)];
    MCMCData = load(MCMCDataDir);
    
    % we only look at the probability with the first index.
    % determine the state of source subjects by the threshold value.
    ProbThreshold = 0.5;
    SourceSubState = zeros(3, 1)+0.1;
    ProbSourceDict = struct();
    for n=1:(N-1)
        SourceName = sprintf('prob_%d', n);
        ProbIter = MCMCData.(SourceName)(:, 1);
        ProbMeanIter = mean(ProbIter);
        ProbLowIter = quantile(ProbIter, 0.05);
        ProbUppIter = quantile(ProbIter, 0.95);
        fprintf('Mean: %.2f, Low: %.2f, Upp: %.2f\n', ProbMeanIter, ProbLowIter, ProbUppIter);
        if ProbLowIter >= ProbThreshold
            SourceSubState(n, 1) = 1;
        elseif ProbUppIter <= ProbThreshold
            SourceSubState(n, 1) = -1;
        else
            SourceSubState(n, 1) = 0;
        end
        ProbSourceDict.(SourceName) = struct('Mean', ProbMeanIter, ...
            'Low', ProbLowIter, 'Upp', ProbUppIter);
    end
    disp(SourceSubState.');

    %% train the swLDA with various combinations of training data (and source subject data)
    MaxVarLen = ceil(SignalLength * 0.4);
    % 1. only the new data
    SignalNewOnly = reshape(permute(SimData.subject_0.X(1:SeqSize, :, :), [3, 2, 1]), ...
        [SignalLength, SeqSize*RCPUnitFlashNum]);
    LabelNewOnly = reshape(permute(SimData.subject_0.Y(1:SeqSize, :), [2, 1]), ...
        [SeqSize*RCPUnitFlashNum, 1]);
    
    [bNewOnly, ~, ~, inModelNewOnly, ~] = trainSWLDAmatlab(SignalNewOnly.', ...
        (LabelNewOnly-0.5)*2, MaxVarLen);
    bNewOnly(inModelNewOnly.' == 0, :) = 0;
    ScoreNewOnly = SignalNewOnly.' * bNewOnly;
    ScoreNewOnlyTar = ScoreNewOnly(LabelNewOnly == 1, :);
    ScoreNewOnlyNtar = ScoreNewOnly(LabelNewOnly ~= 1, :);
    MeanNewOnlyTar = mean(ScoreNewOnlyTar);
    MeanNewOnlyNtar = mean(ScoreNewOnlyNtar);
    StdNewOnly = sqrt(var(ScoreNewOnly));
    
    % 2. new data with strict inclusion of source data
    SignalStrict = SignalNewOnly;
    LabelStrict = LabelNewOnly;
    for n=1:(N-1)
        if SourceSubState(n, 1) == 1
            SourceName = sprintf('subject_%d', n);
            SignalSource_n = reshape(permute(SimData.(SourceName).X, [3, 2, 1]), ...
                [SignalLength, SeqSizeTrain*RCPUnitFlashNum]);
            LabelSource_n = reshape(permute(SimData.(SourceName).Y, [2, 1]), ...
                [SeqSizeTrain*RCPUnitFlashNum, 1]);
            SignalStrict = cat(2, SignalStrict, SignalSource_n);
            LabelStrict = cat(1, LabelStrict, LabelSource_n);
        end
    end
    
    [bStrict, ~, ~, inModelStrict, ~] = trainSWLDAmatlab(SignalStrict.', ...
        (LabelStrict-0.5)*2, MaxVarLen);
    bStrict(inModelStrict.' == 0, :) = 0;
    ScoreStrict = SignalStrict.' * bStrict;
    ScoreStrictTar = ScoreStrict(LabelStrict == 1, :);
    ScoreStrictNtar = ScoreStrict(LabelStrict ~= 1, :);
    MeanStrictTar = mean(ScoreStrictTar);
    MeanStrictNtar = mean(ScoreStrictNtar);
    StdStrict = sqrt(var(ScoreStrict));
    
    % 3. new data with flexible inclusion of source data
    SignalFlexible = SignalStrict;
    LabelFlexible = LabelStrict;
    for n=1:(N-1)
        if SourceSubState(n, 1) == 0
            SourceName = sprintf('subject_%d', n);
            SignalSource_n = reshape(permute(SimData.(SourceName).X, [3, 2, 1]), ...
                [SignalLength, SeqSizeTrain*RCPUnitFlashNum]);
            LabelSource_n = reshape(permute(SimData.(SourceName).Y, [2, 1]), ...
                [SeqSizeTrain*RCPUnitFlashNum, 1]);
            SignalFlexible = cat(2, SignalFlexible, SignalSource_n);
            LabelFlexible = cat(1, LabelFlexible, LabelSource_n);
        end
    end
    
    [bFlexible, ~, ~, inModelFlexible, ~] = trainSWLDAmatlab(SignalFlexible.', ...
        (LabelFlexible-0.5)*2, MaxVarLen);
    bFlexible(inModelFlexible.' == 0, :) = 0;
    ScoreFlexible = SignalFlexible.' * bFlexible;
    ScoreFlexibleTar = ScoreFlexible(LabelFlexible == 1, :);
    ScoreFlexibleNtar = ScoreFlexible(LabelFlexible ~= 1, :);
    MeanFlexibleTar = mean(ScoreFlexibleTar);
    MeanFlexibleNtar = mean(ScoreFlexibleNtar);
    StdFlexible = sqrt(var(ScoreFlexible));
    
    %% save the output
    SimDataswLDADir = [SimDataswLDAFolderDir filesep ...
        sprintf('swLDA_output_seq_size_%d_threshold_%.1f.mat', SeqSize, ProbThreshold)];
    save(SimDataswLDADir, 'ProbSourceDict', 'SourceSubState', 'bNewOnly', ...
        'MeanNewOnlyTar', 'MeanNewOnlyNtar', 'StdNewOnly', ...
        'bStrict', 'MeanStrictTar', 'MeanStrictNtar', 'StdStrict', ...
        'bFlexible', 'MeanFlexibleTar', 'MeanFlexibleNtar', 'StdFlexible', ...
        'ProbThreshold', 'SeqSize');
end
end