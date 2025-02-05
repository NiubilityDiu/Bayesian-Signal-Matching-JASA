clear
clc

%% setup the directory
ParentDir = '/Users/niubilitydiu/Desktop/BSM-Code-V2';
MatlabFunDir = [ParentDir filesep 'MATLAB'];
cd(MatlabFunDir); 

N = 24;
K = 24;
E = 2;
ScenarioName = sprintf('N_%d_K_%d_multi_xdawn_eeg', N, K);
ClusterName = 'borrow_gibbs';

% IterNum = 0;  % start from 0
RCPUnitFlashNum = 12;
SignalLength = 25;
SeqSizeTrain = 10;
TargetLetterSize = 8;
ProbThreshold = 0.1;
LogLhdDiffApprox = '2.0';

% ScenarioName2 = sprintf('%s_multi_option_%d', ScenarioName, OptionID);
SimDataFolderDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'SIM_files' ...
     filesep ScenarioName];


%%
for IterNum=0:0
    fprintf('iter_%d\n', IterNum);
    SimDataFolderDir2 = [SimDataFolderDir filesep ...
        sprintf('iter_%d', IterNum)];
    % create swLDA folder
    SimDataswLDAFolderDir = [SimDataFolderDir2 filesep 'swLDA'];
    mkdir(SimDataswLDAFolderDir);
    
    %% import the training (source data)
    SimDataDir = [SimDataFolderDir2 filesep 'sim_dat.json'];
    SimDataFile = fopen(SimDataDir); % open the file
    SimDataRaw = fread(SimDataFile, inf); % read the contents
    SimDataStr = char(SimDataRaw'); % transform
    fclose(SimDataFile); % close the file
    SimData = jsondecode(jsondecode(jsonencode(SimDataStr)));
    
    %% import the test dataset.
    SimTestDataDir = [SimDataFolderDir2 filesep 'sim_dat_test.json'];
    SimTestDataFile = fopen(SimTestDataDir); % open the file
    SimTestDataRaw = fread(SimTestDataFile, inf); % read the contents
    SimTestDataStr = char(SimTestDataRaw'); % transform
    fclose(SimTestDataFile); % close the file
    SimTestData = jsondecode(jsondecode(jsonencode(SimTestDataStr)));
    clear SimDataRaw SimTestDataRaw
    
    %% import mcmc samples with specified sequence size
    for SeqSize=2:5
        % SeqSize2 = SeqSize * TargetLetterSize;
        fprintf('Seq Size: %d\n', SeqSize);
        MCMCDataDir = [SimDataFolderDir2 filesep ClusterName filesep ...
            sprintf('mcmc_sub_0_seq_size_%d_cluster_log_lhd_diff_approx_%s.mat', SeqSize, LogLhdDiffApprox)];
        MCMCData = load(MCMCDataDir);
        CountMeanIter = mean(MCMCData.chain_1.z_vector, 1);
        CountBinaryIter = (CountMeanIter >= ProbThreshold);     
        disp(CountMeanIter);
        disp(CountBinaryIter);

        %% train the swLDA with various combinations of training data (and source subject data)
        MaxVarLen = ceil(SignalLength * 0.4);
        % 1. only the new data
        SignalNewOnly = reshape(permute(SimData.subject_0.X(:, 1:SeqSize, :, :, :), [5, 4, 3, 2, 1]), ...
            [SignalLength*E, TargetLetterSize*SeqSize*RCPUnitFlashNum]);
        LabelNewOnly = reshape(permute(SimData.subject_0.Y(:, 1:SeqSize, :), [3, 2, 1]), ...
            [TargetLetterSize*SeqSize*RCPUnitFlashNum, 1]);
        
        [bNewOnly, ~, ~, inModelNewOnly, ~] = train_SWLDAmatlab(SignalNewOnly.', ...
            (LabelNewOnly-0.5)*2, MaxVarLen);
        bNewOnly(inModelNewOnly.' == 0, :) = 0;
        ScoreNewOnly = SignalNewOnly.' * bNewOnly;
        ScoreNewOnlyTar = ScoreNewOnly(LabelNewOnly == 1, :);
        ScoreNewOnlyNtar = ScoreNewOnly(LabelNewOnly ~= 1, :);
        MeanNewOnlyTar = mean(ScoreNewOnlyTar);
        MeanNewOnlyNtar = mean(ScoreNewOnlyNtar);
        StdNewOnly = sqrt(var(ScoreNewOnly));
        fprintf('New only: Target Mean: %f, Non-target Mean: %f, Common Std: %f \n', ...
            MeanNewOnlyTar, MeanNewOnlyNtar, StdNewOnly);

        % 2. new data with strict inclusion of source data
        SignalMixture = SignalNewOnly;
        LabelMixture = LabelNewOnly;
        for n=1:(N-1)
            if CountBinaryIter(1, n) == 1
                SourceName = sprintf('subject_%d', n);
                SignalSource_n = reshape(permute(SimData.(SourceName).X, [5, 4, 3, 2, 1]), ...
                    [SignalLength * E, TargetLetterSize*SeqSizeTrain*RCPUnitFlashNum]);
                LabelSource_n = reshape(permute(SimData.(SourceName).Y, [3, 2, 1]), ...
                    [TargetLetterSize*SeqSizeTrain*RCPUnitFlashNum, 1]);
                SignalMixture = cat(2, SignalMixture, SignalSource_n);
                LabelMixture = cat(1, LabelMixture, LabelSource_n);
            end
        end
        
        [bMixture, ~, ~, inModelMixture, ~] = trainSWLDAmatlab(SignalMixture.', ...
            (LabelMixture-0.5)*2, MaxVarLen);
        bMixture(inModelMixture.' == 0, :) = 0;
        ScoreMixture = SignalMixture.' * bMixture;
        ScoreMixtureTar = ScoreMixture(LabelMixture == 1, :);
        ScoreMixtureNtar = ScoreMixture(LabelMixture ~= 1, :);
        MeanMixtureTar = mean(ScoreMixtureTar);
        MeanMixtureNtar = mean(ScoreMixtureNtar);
        StdMixture = sqrt(var(ScoreMixture));
        fprintf('Mixture: Target Mean: %f, Non-target Mean: %f, Common Std: %f \n', ...
            MeanMixtureTar, MeanMixtureNtar, StdMixture);

        %% save the output
        SimDataswLDADir = [SimDataswLDAFolderDir filesep ...
            sprintf('swLDA_output_seq_size_%d.mat', SeqSize)];
        save(SimDataswLDADir, ...
            'bNewOnly', 'MeanNewOnlyTar', 'MeanNewOnlyNtar', 'StdNewOnly', ...
            'bMixture', 'MeanMixtureTar', 'MeanMixtureNtar', 'StdMixture', ...
            'SeqSize');
    end
end

