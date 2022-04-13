clear
clc

%% global constants and environment
UnitFlashNum = 12;
SeqNum = 15;
LetterNumSource = 2;
SampleSizeSource = UnitFlashNum * SeqNum * LetterNumSource;
MaxVarLen = 120;

LocalBool = 0;
RunBool = 0;
switch LocalBool
    case 1
        ParentDir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode';
        SubNewID = 4;
    otherwise 
        ParentDir = '/home/mtianwen';
        SubNewID = str2num(getenv('SLURM_ARRAY_TASK_ID'));
end
SubPool = {'K106', 'K107', 'K108', 'K111', 'K112', ...
    'K113', 'K114', 'K115', 'K117', 'K118', ...
    'K119', 'K120', 'K121', 'K122', 'K123', ...
    'K143', 'K145', 'K146', 'K147', 'K151'};
SubNew = SubPool{SubNewID};
TotalSubNum = 20;
DataType = 'TRN_files';

for LetterNumNew=2:10
    % import the selection matrix
    SelectMatDir = [ParentDir filesep 'EEG_MATLAB_data' filesep DataType ...
        filesep SubNew filesep 'selection_output' filesep ...
        sprintf('source_letter_num_%d', LetterNumSource) filesep ...
        sprintf('select_new_letter_num_%d.mat', LetterNumNew)];
    SelectMat = load(SelectMatDir);
    SelectSub = SelectMat.select_subject;
    SelectSubSize = length(SelectSub);
    if isempty(SelectSub)
        SelectSub = {''};
        SelectSubSize = 0;
    elseif ischar(SelectSub)
        SelectSub = {SelectSub};
        SelectSubSize = 1;
    end
    
    ThresholdValue = SelectMat.threshold;
    
    %% import the new data
    SampleSizeNew = UnitFlashNum * SeqNum * LetterNumNew;
    NewDataDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
                filesep SubNew ...
                filesep sprintf('%s_001_BCI_TRN_Truncated_Data.mat', SubNew)];
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
        
        % save swlda wts obj
        WeightNewOnlyDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
            filesep SubNew filesep 'swLDA' filesep ...
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
    if SelectSubSize >= 1
        for IterId=1:SelectSubSize
            SourceSelectID = SelectSub{IterId};
            disp(SourceSelectID);
            SourceIDDataDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
                filesep SourceSelectID ...
                filesep sprintf('%s_001_BCI_TRN_Truncated_Data.mat', SourceSelectID)];
            SourceSelectIDData = load(SourceIDDataDir);
            SourceSelectIDSignal = SourceSelectIDData.Signal(1:SampleSizeSource,:);
            SourceSelectIDType = SourceSelectIDData.Type(1:SampleSizeSource, :);
            CombineSignal = cat(1, CombineSignal, SourceSelectIDSignal);
            CombineType = cat(1, CombineType, SourceSelectIDType);
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
        
    else
        % use New_Only file
        WeightNewOnlyDir = [ParentDir filesep 'EEG_MATLAB_data' ...
            filesep 'TRN_files' filesep SubNew filesep 'swLDA' filesep ...
            sprintf('Weight_001_BCI_TRN_New_Only_Letter_%d.mat', LetterNumNew)];
        
        WeightNewOnlyMat = load(WeightNewOnlyDir);
        b = WeightNewOnlyMat.b;
        pval = WeightNewOnlyMat.pval;
        inmodel = WeightNewOnlyMat.inmodel;
        MeanTar = WeightNewOnlyMat.MeanTar;
        MeanNtar = WeightNewOnlyMat.MeanNtar;
        StdCommon = WeightNewOnlyMat.StdCommon;
    
    end
    
    % save swlda wts obj
    mkdir([ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
        filesep SubNew filesep 'swLDA' filesep ...
        sprintf('source_letter_num_%d', LetterNumSource)]);
    WeightNewOnlyDir = [ParentDir filesep 'EEG_MATLAB_data' filesep 'TRN_files' ...
        filesep SubNew filesep 'swLDA' filesep ...
        sprintf('source_letter_num_%d', LetterNumSource) filesep ...
        sprintf('Weight_001_BCI_TRN_New_Merged_Letter_%d_KL_Threshold_%s.mat', ...
        LetterNumNew, num2str(ThresholdValue))];
    % make the output consistent
    ID = SubNew;
    sample_size = SampleSizeSource*(TotalSubNum-1) + SampleSizeNew;
    save(WeightNewOnlyDir, 'ID', 'b', 'pval', 'inmodel', 'sample_size', ...
        'MeanTar', 'MeanNtar', 'StdCommon');    
end