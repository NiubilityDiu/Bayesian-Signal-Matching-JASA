clear
clc

%% global constants and environment
UnitFlashNum = 12;
SeqNum = 15;
LetterNumSource = 6;
SampleSizeSource = UnitFlashNum * SeqNum * LetterNumSource;
MaxVarLen = 120;
BinaryThreshold = 0.5;

LocalBool = 1;
RunBool = 0;
switch LocalBool
    case 1
        ParentDir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode';
        LetterNumNew = 3;
    otherwise 
        ParentDir = '/home/mtianwen';
        LetterNumNew = str2num(getenv('SLURM_ARRAY_TASK_ID'));
end

SubPool = [106, 107, 108, 111, 112, 113, 114, 115, 117, 118, ...
    119, 120, 121, 122, 123, 143, 145, 146, 147, 151].';
TotalSubNum = 20;

%%
for NewID=1:TotalSubNum

    SubNew = SubPool(NewID, :);
    SubSource = setdiff(SubPool, SubNew);
    
    % import the selection matrix
    SelectMatDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
        filesep sprintf('Sub_Total_%d', TotalSubNum) filesep sprintf('selection_matrix_letter_num_%d.mat', LetterNumNew)];
    SelectMat = load(SelectMatDir).matrix;
    
    %% import the new data
    SampleSizeNew = UnitFlashNum * SeqNum * LetterNumNew;
    NewDataDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
                filesep sprintf('K%d', SubNew) ...
                filesep sprintf('K%d_001_BCI_TRN_Truncated_Data.mat', SubNew)];
    NewData = load(NewDataDir);
    NewDataSignal = NewData.Signal(1:SampleSizeNew, :);
    NewDataType = NewData.Type(1:SampleSizeNew, :);
    
    %% fit the swLDA with only new data;
    if RunBool
        [b, ~, pval, inmodel, ~] = trainSWLDAmatlab(NewDataSignal, NewDataType, MaxVarLen);
        
        % save the mu1, mu0, and sigma0.
        BInModel = b;
        BInModel(inmodel.' == 0, :) = 0;
        ScoreVec = NewDataSignal * BInModel;
        ScoreVecTar = ScoreVec(NewDataType == 1, :);
        ScoreVecNtar = ScoreVec(NewDataType ~= 1, :);
        MeanTar = mean(ScoreVecTar);
        MeanNtar = mean(ScoreVecNtar);
        StdCommon = sqrt(var(ScoreVec));
        
        disp('Only New Data:')
        disp(MeanTar);
        disp(MeanNtar);
        disp(StdCommon);
        
        %%
        % save swlda wts obj
        WeightNewOnlyDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
            filesep sprintf('K%d', SubNew) filesep 'swLDA' filesep ...
            sprintf('Weight_001_BCI_TRN_New_Only_Letter_%d.mat', LetterNumNew)];
        % make the output consistent
        ID = SubNew;
        sample_size = SampleSizeNew;
        save(WeightNewOnlyDir, 'ID', 'b', 'pval', 'inmodel', 'sample_size', ...
            'MeanTar', 'MeanNtar', 'StdCommon');
    end

    %% fit the swLDA with combined data;
    CombineSignal = NewDataSignal;
    CombineType = NewDataType;
    for SourceID=1:TotalSubNum
        if SelectMat(NewID, SourceID) < 1 && SelectMat(NewID, SourceID) > BinaryThreshold
            disp(SourceID);
            SourceIDDataDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
                filesep sprintf('K%d', SubPool(SourceID,:)) ...
                filesep sprintf('K%d_001_BCI_TRN_Truncated_Data.mat', SubPool(SourceID,:))];
            SourceIDData = load(SourceIDDataDir);
            SourceIDSignal = SourceIDData.Signal(1:SampleSizeSource,:);
            SourceIDType = SourceIDData.Type(1:SampleSizeSource, :);
            CombineSignal = cat(1, CombineSignal, SourceIDSignal);
            CombineType = cat(1, CombineType, SourceIDType);
        end
    end
    
    % fit the swLDA with only combined data;
    [b, ~, pval, inmodel, ~] = trainSWLDAmatlab(CombineSignal, CombineType, MaxVarLen);
    
    % save the mu1, mu0, and sigma0.
    BInModel = b;
    BInModel(inmodel.' == 0, :) = 0;
    ScoreVec = NewDataSignal * BInModel;
    ScoreVecTar = ScoreVec(NewDataType == 1, :);
    ScoreVecNtar = ScoreVec(NewDataType ~= 1, :);
    MeanTar = mean(ScoreVecTar);
    MeanNtar = mean(ScoreVecNtar);
    StdCommon = sqrt(var(ScoreVec));
    
    disp('New + Selected Source Data:')
    disp(MeanTar);
    disp(MeanNtar);
    disp(StdCommon);
    
    % save swlda wts obj
    WeightNewOnlyDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
        filesep sprintf('K%d', SubNew) filesep 'swLDA' filesep ...
        sprintf('Weight_001_BCI_TRN_New_Merged_Letter_%d_%s.mat', ...
        LetterNumNew, num2str(BinaryThreshold))];
    % make the output consistent
    ID = SubNew;
    sample_size = SampleSizeSource*(TotalSubNum-1) + SampleSizeNew;
    save(WeightNewOnlyDir, 'ID', 'b', 'pval', 'inmodel', 'sample_size', ...
        'MeanTar', 'MeanNtar', 'StdCommon');

end