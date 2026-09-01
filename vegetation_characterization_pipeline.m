%% ========================================================================
% REMOTE VEGETATION CHARACTERIZATION USING DEEP LEARNING
% Academic Project - Master Pipeline
%
% Objective
% ---------
% Develop a remote vegetation characterization workflow for a study area
% where an on-site vegetation survey was not available.
%
% The project integrates:
%
%   - Historical terrestrial imagery
%   - Public image databases and APIs
%   - Computer Vision
%   - ResNet-18 and Transfer Learning
%   - Vegetation cover classification
%   - Closed-set probable species classification
%   - Landsat and Sentinel-2 satellite imagery
%   - NDVI-based vegetation analysis
%
% Validated project results
% -------------------------
% Balanced vegetation dataset:        1,092 images
% Vegetation classes:                 4
% Independent vegetation test:        164 images
% Vegetation classifier accuracy:     75.00 %
%
% Closed-set species classes:         8
% Species classifier accuracy:        79.55 %
%
% Historical observation points:      9
% Historical terrestrial images:      54
%
% IMPORTANT
% ---------
% Species outputs must be interpreted as probable identifications.
% The species model is a closed-set classifier and does not replace an
% on-site botanical survey.
%
% Historical results stored in this script are explicitly identified as
% archived validated project results. They are not presented as if they
% had been recomputed during the current execution.
%
% MATLAB R2026a
% ========================================================================

clc;
clear;
close all;

rng(42);

%% ========================================================================
% 1. PROJECT ROOT
% ========================================================================

scriptFile = mfilename('fullpath');

if isempty(scriptFile)
    scriptFolder = pwd;
else
    scriptFolder = fileparts(scriptFile);
end

[~,currentFolderName] = fileparts(scriptFolder);

% Allows this file to be placed either in the repository root or in /src.
if strcmpi(currentFolderName,'src')
    projectRoot = fileparts(scriptFolder);
else
    projectRoot = scriptFolder;
end

fprintf('\n');
fprintf('============================================================\n');
fprintf('REMOTE VEGETATION CHARACTERIZATION PROJECT\n');
fprintf('============================================================\n');
fprintf('Project root:\n%s\n',projectRoot);


%% ========================================================================
% 2. EXECUTION CONFIGURATION
% ========================================================================

% Keep these options FALSE when you only want to reproduce the GitHub
% figures and export the validated results obtained during the project.
%
% Set them to TRUE only when the corresponding raw dataset is available
% and you intentionally want to train a new model.

config.trainVegetationModel = false;
config.trainSpeciesModel    = false;
config.queryINaturalist     = false;

config.removeExactDuplicates = true;

config.trainingPlots = "training-progress";

% Species data are not automatically balanced because the original
% species bank contained different numbers of images for each class.
config.balanceSpeciesDataset = false;


%% ========================================================================
% 3. PROJECT FOLDERS
% ========================================================================

folders.data = fullfile(projectRoot,'data');
folders.models = fullfile(projectRoot,'models');
folders.results = fullfile(projectRoot,'results');
folders.figures = fullfile(projectRoot,'images');

ensureFolder(folders.data);
ensureFolder(folders.models);
ensureFolder(folders.results);
ensureFolder(folders.figures);


%% ========================================================================
% 4. STUDY AREA
% ========================================================================

study.name = "Jardim Guaruja - Sao Paulo, Brazil";

study.latitude  = -23.70025;
study.longitude = -46.79275;

study.iNaturalistRadiusKm = 10;


%% ========================================================================
% 5. DATA SOURCES USED IN THE PROJECT
% ========================================================================

% Wikimedia Commons and Flickr were used during dataset construction.
% This master script documents these sources but does not automatically
% re-download the complete image dataset.

sourceName = [
    "Google Street View"
    "iNaturalist"
    "Wikimedia Commons"
    "Flickr"
    "Landsat 5"
    "Sentinel-2"
];

sourcePurpose = [
    "Historical terrestrial imagery"
    "Regional vegetation occurrence evidence"
    "Training image source"
    "Training image source"
    "Historical NDVI analysis"
    "Recent NDVI analysis"
];

dataSources = table( ...
    sourceName, ...
    sourcePurpose, ...
    'VariableNames',{'Source','Purpose'});

writetable( ...
    dataSources, ...
    fullfile(folders.results,'data_sources.csv'));

fprintf('\nData sources documented.\n');


%% ========================================================================
% 6. OPTIONAL iNATURALIST REGIONAL QUERY
% ========================================================================

% iNaturalist data are used only as regional occurrence evidence.
%
% A nearby record does NOT prove that a species occurs inside the study
% polygon.

if config.queryINaturalist

    fprintf('\nQuerying nearby plant species from iNaturalist...\n');

    try

        regionalSpecies = fetchINaturalistPlants( ...
            study.latitude, ...
            study.longitude, ...
            study.iNaturalistRadiusKm);

        writetable( ...
            regionalSpecies, ...
            fullfile( ...
            folders.results, ...
            'regional_species_inaturalist.csv'));

        disp(regionalSpecies(1:min(20,height(regionalSpecies)),:));

    catch ME

        warning( ...
            'iNaturalist query could not be completed: %s', ...
            ME.message);

    end

end


%% ========================================================================
% 7. VEGETATION DATASET LOCATION
% ========================================================================

% Original local project structure:
%
% IA_Vegetacao/
%   imagens/
%       arvore/
%       arbusto/
%       vegetacao_rasteira/
%       solo_exposto/
%
% Recommended GitHub structure:
%
% data/
%   vegetation_dataset/
%       arvore/
%       arbusto/
%       vegetacao_rasteira/
%       solo_exposto/

vegetationDatasetCandidates = {
    fullfile(projectRoot,'data','vegetation_dataset')
    fullfile(projectRoot,'imagens')
};

vegetationDatasetFolder = ...
    firstExistingFolder(vegetationDatasetCandidates);

vegetationDatasetAvailable = ...
    strlength(vegetationDatasetFolder) > 0;


%% ========================================================================
% 8. ARCHIVED VALIDATED VEGETATION RESULTS
% ========================================================================

% These values come from the validated independent test performed during
% development of the project.
%
% Rows = true classes
% Columns = predicted classes
%
% Class order:
% Shrub
% Tree
% Bare soil
% Ground vegetation

archivedVegetation.classKeys = [
    "arbusto"
    "arvore"
    "solo_exposto"
    "vegetacao_rasteira"
];

archivedVegetation.displayNames = [
    "Shrub"
    "Tree"
    "Bare soil"
    "Ground vegetation"
];

archivedVegetation.confusionMatrix = [
    29  6  0  6
     4 36  0  1
     1 16 24  0
     2  4  1 34
];

archivedVegetation.datasetSize = 1092;
archivedVegetation.trainingSize = 764;
archivedVegetation.validationSize = 164;
archivedVegetation.testSize = 164;

archivedVegetation.accuracy = ...
    100 * ...
    sum(diag(archivedVegetation.confusionMatrix)) / ...
    sum(archivedVegetation.confusionMatrix,'all');


%% ========================================================================
% 9. LOAD AND PREPARE VEGETATION DATASET
% ========================================================================

vegetationData = [];
vegetationTrain = [];
vegetationValidation = [];
vegetationTest = [];

if vegetationDatasetAvailable

    fprintf('\nVegetation dataset found:\n%s\n', ...
        vegetationDatasetFolder);

    vegetationData = imageDatastore( ...
        vegetationDatasetFolder, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

    fprintf('\nOriginal dataset distribution:\n');

    disp(countEachLabel(vegetationData));

    if config.removeExactDuplicates

        fprintf('Checking exact file duplicates...\n');

        vegetationData = ...
            removeExactDuplicateImages(vegetationData);

    end

    validateVegetationClasses( ...
        vegetationData.Labels, ...
        archivedVegetation.classKeys);

    vegetationData = ...
        balanceImageDatastore(vegetationData);

    fprintf('\nBalanced dataset distribution:\n');

    disp(countEachLabel(vegetationData));

    fprintf('Balanced dataset size: %d images\n', ...
        numel(vegetationData.Files));

    % 70% training, 15% validation, 15% independent test.
    [vegetationTrain, ...
     vegetationValidation, ...
     vegetationTest] = splitEachLabel( ...
        vegetationData, ...
        0.70, ...
        0.15, ...
        "randomized");

    fprintf('\nCurrent split:\n');
    fprintf('Training:   %d images\n',numel(vegetationTrain.Files));
    fprintf('Validation: %d images\n',numel(vegetationValidation.Files));
    fprintf('Test:       %d images\n',numel(vegetationTest.Files));

else

    fprintf('\nRaw vegetation dataset was not found.\n');
    fprintf(['Validated archived results will be used for figures ' ...
             'and project documentation.\n']);

end


%% ========================================================================
% 10. VEGETATION CLASSIFIER - RESNET-18
% ========================================================================

vegetationNet = [];

vegetationResultSource = ...
    "Archived validated project experiment";

vegetationConfusionMatrix = ...
    archivedVegetation.confusionMatrix;

vegetationAccuracy = ...
    archivedVegetation.accuracy;

vegetationDatasetSize = ...
    archivedVegetation.datasetSize;

vegetationTrainingSize = ...
    archivedVegetation.trainingSize;

vegetationValidationSize = ...
    archivedVegetation.validationSize;

vegetationTestSize = ...
    archivedVegetation.testSize;


if config.trainVegetationModel

    if ~vegetationDatasetAvailable

        error([ ...
            'Vegetation training was requested, but the raw vegetation ' ...
            'dataset was not found.']);

    end

    fprintf('\n============================================================\n');
    fprintf('TRAINING VEGETATION CLASSIFIER\n');
    fprintf('============================================================\n');

    vegetationClassNames = ...
        string(categories(vegetationTrain.Labels));

    numberOfVegetationClasses = ...
        numel(vegetationClassNames);

    % Modern MATLAB transfer-learning workflow.
    vegetationNet = imagePretrainedNetwork( ...
        "resnet18", ...
        NumClasses=numberOfVegetationClasses);

    vegetationInputSize = ...
        vegetationNet.Layers(1).InputSize;

    augmenter = imageDataAugmenter( ...
        RandXReflection=true, ...
        RandXTranslation=[-10 10], ...
        RandYTranslation=[-10 10], ...
        RandScale=[0.90 1.10]);

    augmentedVegetationTrain = augmentedImageDatastore( ...
        vegetationInputSize(1:2), ...
        vegetationTrain, ...
        DataAugmentation=augmenter, ...
        ColorPreprocessing="gray2rgb");

    augmentedVegetationValidation = augmentedImageDatastore( ...
        vegetationInputSize(1:2), ...
        vegetationValidation, ...
        ColorPreprocessing="gray2rgb");

    augmentedVegetationTest = augmentedImageDatastore( ...
        vegetationInputSize(1:2), ...
        vegetationTest, ...
        ColorPreprocessing="gray2rgb");

    validationFrequency = max( ...
        1, ...
        floor(numel(vegetationTrain.Files)/32));

    vegetationOptions = trainingOptions( ...
        "adam", ...
        InitialLearnRate=1e-4, ...
        MiniBatchSize=32, ...
        MaxEpochs=10, ...
        Shuffle="every-epoch", ...
        ValidationData=augmentedVegetationValidation, ...
        ValidationFrequency=validationFrequency, ...
        Metrics="accuracy", ...
        Plots=config.trainingPlots, ...
        Verbose=false, ...
        ExecutionEnvironment="auto");

    vegetationNet = trainnet( ...
        augmentedVegetationTrain, ...
        vegetationNet, ...
        "crossentropy", ...
        vegetationOptions);

    fprintf('\nEvaluating the independent test set...\n');

    vegetationScores = minibatchpredict( ...
        vegetationNet, ...
        augmentedVegetationTest);

    vegetationPredictions = scores2label( ...
        vegetationScores, ...
        vegetationClassNames);

    vegetationConfusionMatrix = buildVegetationConfusionMatrix( ...
        vegetationTest.Labels, ...
        vegetationPredictions, ...
        archivedVegetation.classKeys);

    vegetationAccuracy = ...
        100 * ...
        sum(diag(vegetationConfusionMatrix)) / ...
        sum(vegetationConfusionMatrix,'all');

    vegetationDatasetSize = ...
        numel(vegetationData.Files);

    vegetationTrainingSize = ...
        numel(vegetationTrain.Files);

    vegetationValidationSize = ...
        numel(vegetationValidation.Files);

    vegetationTestSize = ...
        numel(vegetationTest.Files);

    vegetationResultSource = ...
        "Recomputed during the current execution";

    save( ...
        fullfile( ...
        folders.models, ...
        'vegetation_resnet18.mat'), ...
        'vegetationNet', ...
        'vegetationClassNames', ...
        'vegetationInputSize', ...
        '-v7.3');

end


%% ========================================================================
% 11. VEGETATION PERFORMANCE METRICS
% ========================================================================

vegetationMetrics = ...
    metricsFromConfusionMatrix( ...
        vegetationConfusionMatrix, ...
        archivedVegetation.displayNames);

fprintf('\nVegetation classifier results:\n');
fprintf('Accuracy: %.2f %%\n',vegetationAccuracy);
fprintf('Result source: %s\n',vegetationResultSource);

disp(vegetationMetrics);

writetable( ...
    vegetationMetrics, ...
    fullfile( ...
    folders.results, ...
    'vegetation_classifier_metrics.csv'));

writematrix( ...
    vegetationConfusionMatrix, ...
    fullfile( ...
    folders.results, ...
    'vegetation_confusion_matrix.csv'));


%% ========================================================================
% 12. VEGETATION CONFUSION MATRIX FIGURE
% ========================================================================

figConfusion = figure( ...
    'Color','w', ...
    'Position',[100 100 950 800]);

imagesc(vegetationConfusionMatrix);

axis equal;
axis tight;

colormap(parula);

colorBar = colorbar;
colorBar.Label.String = 'Number of samples';

xticks(1:4);
yticks(1:4);

xticklabels(archivedVegetation.displayNames);
yticklabels(archivedVegetation.displayNames);

xlabel( ...
    'Predicted class', ...
    'FontWeight','bold');

ylabel( ...
    'True class', ...
    'FontWeight','bold');

title( ...
    'Confusion Matrix - Vegetation Classifier', ...
    'FontSize',17, ...
    'FontWeight','bold');

maximumCellValue = ...
    max(vegetationConfusionMatrix(:));

for row = 1:size(vegetationConfusionMatrix,1)

    for column = 1:size(vegetationConfusionMatrix,2)

        currentValue = ...
            vegetationConfusionMatrix(row,column);

        if currentValue > maximumCellValue/2
            currentColor = 'w';
        else
            currentColor = 'k';
        end

        text( ...
            column, ...
            row, ...
            num2str(currentValue), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',14, ...
            'FontWeight','bold', ...
            'Color',currentColor);

    end

end

exportgraphics( ...
    figConfusion, ...
    fullfile( ...
    folders.figures, ...
    'confusion_matrix.png'), ...
    'Resolution',300);

close(figConfusion);


%% ========================================================================
% 13. ARCHIVED CLOSED-SET SPECIES CLASSIFIER RESULTS
% ========================================================================

% The complementary species classifier contains eight classes only.
%
% It must NOT be interpreted as an open-world botanical identification
% system.
%
% Plants outside these eight classes can still receive one of these labels,
% which is why predictions must be interpreted as probable candidates.

archivedSpecies.names = [
    "Araucaria angustifolia"
    "Celtis chichape"
    "Celtis clausseniana"
    "Eriobotrya japonica"
    "Erythrina speciosa"
    "Eugenia uniflora"
    "Euterpe edulis"
    "Schinus terebinthifolia"
];

archivedSpecies.precision = [
    100.000
     80.000
     70.000
     78.571
     90.000
     87.500
    100.000
     50.000
];

archivedSpecies.recall = [
     90.909
     72.727
     63.636
    100.000
     81.818
     63.636
     90.909
     72.727
];

archivedSpecies.f1 = [
    95.238
    76.190
    66.667
    88.000
    85.714
    73.684
    95.238
    59.259
];

archivedSpecies.accuracy = 79.55;

speciesAccuracy = ...
    archivedSpecies.accuracy;

speciesResultSource = ...
    "Archived validated project experiment";

speciesMetrics = table( ...
    archivedSpecies.names, ...
    archivedSpecies.precision, ...
    archivedSpecies.recall, ...
    archivedSpecies.f1, ...
    'VariableNames',{ ...
        'Species', ...
        'Precision_percent', ...
        'Recall_percent', ...
        'F1_percent'});


%% ========================================================================
% 14. OPTIONAL SPECIES MODEL RETRAINING
% ========================================================================

speciesDatasetCandidates = {
    fullfile(projectRoot,'data','species_dataset')
    fullfile(projectRoot,'imagens_especies')
};

speciesDatasetFolder = ...
    firstExistingFolder(speciesDatasetCandidates);

speciesDatasetAvailable = ...
    strlength(speciesDatasetFolder) > 0;


if config.trainSpeciesModel

    if ~speciesDatasetAvailable

        error([ ...
            'Species training was requested, but a species dataset folder ' ...
            'was not found.']);

    end

    fprintf('\n============================================================\n');
    fprintf('TRAINING CLOSED-SET SPECIES CLASSIFIER\n');
    fprintf('============================================================\n');

    speciesData = imageDatastore( ...
        speciesDatasetFolder, ...
        IncludeSubfolders=true, ...
        LabelSource="foldernames");

    if config.removeExactDuplicates

        speciesData = ...
            removeExactDuplicateImages(speciesData);

    end

    if config.balanceSpeciesDataset

        speciesData = ...
            balanceImageDatastore(speciesData);

    end

    [speciesTrain, ...
     speciesValidation, ...
     speciesTest] = splitEachLabel( ...
        speciesData, ...
        0.70, ...
        0.15, ...
        "randomized");

    speciesClassNames = ...
        string(categories(speciesTrain.Labels));

    numberOfSpecies = ...
        numel(speciesClassNames);

    speciesNet = imagePretrainedNetwork( ...
        "resnet18", ...
        NumClasses=numberOfSpecies);

    speciesInputSize = ...
        speciesNet.Layers(1).InputSize;

    speciesAugmenter = imageDataAugmenter( ...
        RandXReflection=true, ...
        RandXTranslation=[-10 10], ...
        RandYTranslation=[-10 10], ...
        RandScale=[0.90 1.10]);

    augmentedSpeciesTrain = augmentedImageDatastore( ...
        speciesInputSize(1:2), ...
        speciesTrain, ...
        DataAugmentation=speciesAugmenter, ...
        ColorPreprocessing="gray2rgb");

    augmentedSpeciesValidation = augmentedImageDatastore( ...
        speciesInputSize(1:2), ...
        speciesValidation, ...
        ColorPreprocessing="gray2rgb");

    augmentedSpeciesTest = augmentedImageDatastore( ...
        speciesInputSize(1:2), ...
        speciesTest, ...
        ColorPreprocessing="gray2rgb");

    speciesValidationFrequency = max( ...
        1, ...
        floor(numel(speciesTrain.Files)/32));

    speciesOptions = trainingOptions( ...
        "adam", ...
        InitialLearnRate=1e-4, ...
        MiniBatchSize=32, ...
        MaxEpochs=10, ...
        Shuffle="every-epoch", ...
        ValidationData=augmentedSpeciesValidation, ...
        ValidationFrequency=speciesValidationFrequency, ...
        Metrics="accuracy", ...
        Plots=config.trainingPlots, ...
        Verbose=false, ...
        ExecutionEnvironment="auto");

    speciesNet = trainnet( ...
        augmentedSpeciesTrain, ...
        speciesNet, ...
        "crossentropy", ...
        speciesOptions);

    speciesScores = minibatchpredict( ...
        speciesNet, ...
        augmentedSpeciesTest);

    speciesPredictions = scores2label( ...
        speciesScores, ...
        speciesClassNames);

    speciesTrueLabels = ...
        speciesTest.Labels;

    speciesConfusionMatrix = buildGenericConfusionMatrix( ...
        speciesTrueLabels, ...
        speciesPredictions, ...
        speciesClassNames);

    speciesAccuracy = ...
        100 * ...
        sum(diag(speciesConfusionMatrix)) / ...
        sum(speciesConfusionMatrix,'all');

    speciesMetrics = ...
        metricsFromConfusionMatrix( ...
            speciesConfusionMatrix, ...
            speciesClassNames);

    speciesResultSource = ...
        "Recomputed during the current execution";

    save( ...
        fullfile( ...
        folders.models, ...
        'species_resnet18_closed_set.mat'), ...
        'speciesNet', ...
        'speciesClassNames', ...
        'speciesInputSize', ...
        '-v7.3');

end


fprintf('\nClosed-set species classifier results:\n');
fprintf('Accuracy: %.2f %%\n',speciesAccuracy);
fprintf('Result source: %s\n',speciesResultSource);

writetable( ...
    speciesMetrics, ...
    fullfile( ...
    folders.results, ...
    'species_classifier_metrics.csv'));


%% ========================================================================
% 15. HISTORICAL TERRESTRIAL IMAGE INVENTORY
% ========================================================================

% Historical image structure used in the project:
%
% SERIEHSIT/
%   PONTO 0/
%   PONTO 1/
%   ...
%   PONTO 8/
%
% Years:
%
% 2010
% 2011
% 2018
% 2019
% 2022
% 2024
%
% Expected total:
%
% 9 observation points x 6 years = 54 valid records.

historicalFolderCandidates = {
    fullfile(projectRoot,'data','historical_images')
    fullfile(projectRoot,'SERIEHSIT')
};

historicalFolder = ...
    firstExistingFolder(historicalFolderCandidates);

historicalImageCount = 54;

historicalResultSource = ...
    "Archived validated project inventory";


if strlength(historicalFolder) > 0

    historicalInventory = ...
        buildHistoricalImageInventory(historicalFolder);

    historicalImageCount = ...
        height(historicalInventory);

    historicalResultSource = ...
        "Inventory generated during the current execution";

    fprintf('\nHistorical image inventory:\n');
    fprintf('%d valid historical records found.\n', ...
        historicalImageCount);

    if historicalImageCount ~= 54

        warning([ ...
            'The expected project inventory contains 54 images, but %d ' ...
            'valid historical records were found.'], ...
            historicalImageCount);

    end

    writetable( ...
        historicalInventory, ...
        fullfile( ...
        folders.results, ...
        'historical_image_inventory.csv'));

else

    fprintf('\nHistorical image folder was not found.\n');
    fprintf('Archived validated count: 54 terrestrial images.\n');

end


%% ========================================================================
% 16. VALIDATED HISTORICAL NDVI RESULTS
% ========================================================================

% These values are stored as validated outputs from the satellite analysis.
%
% They are NOT recalculated by this execution unless the raw satellite
% preprocessing pipeline is independently reconstructed.
%
% Sensor note:
%
% 2010 and 2011 -> Landsat 5
% 2018 onward   -> Sentinel-2
%
% Because the sensors have different spatial resolutions, the time series
% should be interpreted as a broad vegetation trend rather than a perfectly
% homogeneous sensor-to-sensor comparison.

ndviYear = [
    2010
    2011
    2018
    2019
    2022
    2024
    2026
];

ndviSensor = [
    "Landsat 5"
    "Landsat 5"
    "Sentinel-2"
    "Sentinel-2"
    "Sentinel-2"
    "Sentinel-2"
    "Sentinel-2"
];

meanNDVI = [
    0.35630
    0.39623
    0.43925
    0.44651
    0.41351
    0.46087
    0.48402
];

medianNDVI = [
    0.33602
    0.36996
    NaN
    NaN
    NaN
    0.49544
    0.51129
];

denseVegetationPercent = [
     0.000
     8.333
    30.631
    30.631
    26.126
    36.036
    41.441
];

ndviResults = table( ...
    ndviYear, ...
    ndviSensor, ...
    meanNDVI, ...
    medianNDVI, ...
    denseVegetationPercent, ...
    'VariableNames',{ ...
        'Year', ...
        'Sensor', ...
        'Mean_NDVI', ...
        'Median_NDVI', ...
        'Dense_vegetation_percent'});

writetable( ...
    ndviResults, ...
    fullfile( ...
    folders.results, ...
    'ndvi_time_series.csv'));

fprintf('\nValidated historical NDVI results:\n');

disp(ndviResults);


%% ========================================================================
% 17. NDVI TREND
% ========================================================================

ndviChange2010to2026 = ...
    100 * ...
    (meanNDVI(end)-meanNDVI(1)) / ...
    meanNDVI(1);

fprintf('NDVI change from 2010 to 2026: %.1f %%\n', ...
    ndviChange2010to2026);

fprintf('2024 mean NDVI: %.3f\n', ...
    meanNDVI(ndviYear == 2024));

fprintf('2024 dense vegetation: %.1f %%\n', ...
    denseVegetationPercent(ndviYear == 2024));

fprintf('2026 mean NDVI: %.3f\n', ...
    meanNDVI(ndviYear == 2026));

fprintf('2026 dense vegetation: %.1f %%\n', ...
    denseVegetationPercent(ndviYear == 2026));


%% ========================================================================
% 18. NDVI FIGURE
% ========================================================================

figNDVI = figure( ...
    'Color','w', ...
    'Position',[100 100 1200 680]);

plot( ...
    ndviYear, ...
    meanNDVI, ...
    '-o', ...
    'LineWidth',2.5, ...
    'MarkerSize',8);

grid on;
box off;

xlabel( ...
    'Year', ...
    'FontWeight','bold');

ylabel( ...
    'Mean NDVI', ...
    'FontWeight','bold');

title( ...
    'Temporal Evolution of Mean NDVI', ...
    'FontSize',18, ...
    'FontWeight','bold');

ylim([0.30 0.53]);
xlim([2009.5 2026.5]);

for i = 1:numel(ndviYear)

    text( ...
        ndviYear(i), ...
        meanNDVI(i)+0.008, ...
        sprintf('%.3f',meanNDVI(i)), ...
        'HorizontalAlignment','center', ...
        'FontSize',10);

end

text( ...
    2011, ...
    0.315, ...
    'Landsat 5 period', ...
    'FontSize',10);

text( ...
    2022, ...
    0.315, ...
    'Sentinel-2 period', ...
    'FontSize',10);

exportgraphics( ...
    figNDVI, ...
    fullfile( ...
    folders.figures, ...
    'ndvi_analysis.png'), ...
    'Resolution',300);

close(figNDVI);


%% ========================================================================
% 19. PROJECT WORKFLOW FIGURE
% ========================================================================

figWorkflow = figure( ...
    'Color','w', ...
    'Position',[100 100 1600 700]);

workflowAxes = axes(figWorkflow);

axis(workflowAxes,[0 1 0 1]);
axis(workflowAxes,'off');

hold(workflowAxes,'on');

text( ...
    workflowAxes, ...
    0.5, ...
    0.93, ...
    'Remote Vegetation Characterization Workflow', ...
    'HorizontalAlignment','center', ...
    'FontSize',20, ...
    'FontWeight','bold');

text( ...
    workflowAxes, ...
    0.5, ...
    0.875, ...
    'Integration of public data, computer vision and remote sensing', ...
    'HorizontalAlignment','center', ...
    'FontSize',11);

workflowBoxWidth = 0.16;
workflowBoxHeight = 0.30;

workflowX = [
    0.03
    0.225
    0.420
    0.615
    0.810
];

workflowY = 0.42;

workflowTitles = {
    '1. DATA SOURCES'
    '2. PREPARATION'
    '3. DEEP LEARNING'
    '4. REMOTE SENSING'
    '5. INTERPRETATION'
};

workflowContent = {
    {'Street View','iNaturalist','Wikimedia / Flickr'}
    {'Cleaning','Duplicate removal','Dataset balancing'}
    {'ResNet-18','Transfer Learning','Image classification'}
    {'Landsat / Sentinel-2','NDVI','Temporal assessment'}
    {'Vegetation cover','Probable species','Integrated results'}
};

workflowMetrics = {
    '54 historical images'
    '1,092 balanced images'
    '75.0% test accuracy'
    'Satellite imagery + NDVI'
    'Remote characterization'
};

for i = 1:5

    rectangle( ...
        workflowAxes, ...
        'Position',[ ...
            workflowX(i), ...
            workflowY, ...
            workflowBoxWidth, ...
            workflowBoxHeight], ...
        'Curvature',0.05, ...
        'LineWidth',1.5, ...
        'EdgeColor',[0.20 0.20 0.20], ...
        'FaceColor',[0.97 0.97 0.97]);

    text( ...
        workflowAxes, ...
        workflowX(i)+workflowBoxWidth/2, ...
        workflowY+0.245, ...
        workflowTitles{i}, ...
        'HorizontalAlignment','center', ...
        'FontSize',10.3, ...
        'FontWeight','bold');

    plot( ...
        workflowAxes, ...
        [ ...
            workflowX(i)+0.015 ...
            workflowX(i)+workflowBoxWidth-0.015], ...
        [ ...
            workflowY+0.210 ...
            workflowY+0.210], ...
        'LineWidth',1);

    text( ...
        workflowAxes, ...
        workflowX(i)+workflowBoxWidth/2, ...
        workflowY+0.110, ...
        workflowContent{i}, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontSize',9.3);

    text( ...
        workflowAxes, ...
        workflowX(i)+workflowBoxWidth/2, ...
        workflowY-0.050, ...
        workflowMetrics{i}, ...
        'HorizontalAlignment','center', ...
        'FontSize',9.3, ...
        'FontWeight','bold');

end

for i = 1:4

    arrowStart = ...
        workflowX(i)+workflowBoxWidth+0.008;

    arrowEnd = ...
        workflowX(i+1)-0.008;

    quiver( ...
        workflowAxes, ...
        arrowStart, ...
        workflowY+workflowBoxHeight/2, ...
        arrowEnd-arrowStart, ...
        0, ...
        0, ...
        'LineWidth',1.5, ...
        'MaxHeadSize',0.9, ...
        'Color',[0.2 0.2 0.2]);

end

text( ...
    workflowAxes, ...
    0.5, ...
    0.18, ...
    ['Objective: characterize vegetation in a study area ' ...
     'without an on-site survey'], ...
    'HorizontalAlignment','center', ...
    'FontSize',12, ...
    'FontWeight','bold');

exportgraphics( ...
    figWorkflow, ...
    fullfile( ...
    folders.figures, ...
    'project_workflow.png'), ...
    'Resolution',300);

close(figWorkflow);


%% ========================================================================
% 20. PROJECT RESULTS FIGURE
% ========================================================================

figResults = figure( ...
    'Color','w', ...
    'Position',[100 100 1450 720]);

resultsAxes = axes(figResults);

axis(resultsAxes,[0 1 0 1]);
axis(resultsAxes,'off');

hold(resultsAxes,'on');

text( ...
    resultsAxes, ...
    0.5, ...
    0.91, ...
    'Project Results Summary', ...
    'HorizontalAlignment','center', ...
    'FontSize',22, ...
    'FontWeight','bold');

text( ...
    resultsAxes, ...
    0.5, ...
    0.855, ...
    ['Main quantitative results from the remote vegetation ' ...
     'characterization workflow'], ...
    'HorizontalAlignment','center', ...
    'FontSize',11);

resultCardX = [
    0.055
    0.285
    0.515
    0.745
];

resultValues = {
    sprintf('%d',vegetationDatasetSize)
    sprintf('%.1f%%',vegetationAccuracy)
    sprintf('%.2f%%',speciesAccuracy)
    sprintf('%d',historicalImageCount)
};

resultTitles = {
    'Balanced Dataset'
    'Vegetation Classifier'
    'Closed-Set Species Model'
    'Historical Imagery'
};

resultDescriptions = {
    'images across four vegetation classes'
    sprintf('%d-image independent test set',vegetationTestSize)
    'eight candidate species classes'
    'images from 2010 to 2024'
};

for i = 1:4

    rectangle( ...
        resultsAxes, ...
        'Position',[ ...
            resultCardX(i), ...
            0.41, ...
            0.20, ...
            0.29], ...
        'Curvature',0.05, ...
        'LineWidth',1.5, ...
        'EdgeColor',[0.20 0.20 0.20], ...
        'FaceColor',[0.97 0.97 0.97]);

    text( ...
        resultsAxes, ...
        resultCardX(i)+0.10, ...
        0.595, ...
        resultValues{i}, ...
        'HorizontalAlignment','center', ...
        'FontSize',24, ...
        'FontWeight','bold');

    text( ...
        resultsAxes, ...
        resultCardX(i)+0.10, ...
        0.515, ...
        resultTitles{i}, ...
        'HorizontalAlignment','center', ...
        'FontSize',10.3, ...
        'FontWeight','bold');

    text( ...
        resultsAxes, ...
        resultCardX(i)+0.10, ...
        0.455, ...
        resultDescriptions{i}, ...
        'HorizontalAlignment','center', ...
        'FontSize',9.0);

end

text( ...
    resultsAxes, ...
    0.5, ...
    0.29, ...
    ['MATLAB | ResNet-18 | Transfer Learning | Computer Vision | ' ...
     'Remote Sensing | NDVI'], ...
    'HorizontalAlignment','center', ...
    'FontSize',11.2, ...
    'FontWeight','bold');

text( ...
    resultsAxes, ...
    0.5, ...
    0.22, ...
    ['Landsat | Sentinel-2 | Historical Imagery | ' ...
     'Public APIs and Image Databases'], ...
    'HorizontalAlignment','center', ...
    'FontSize',10.2);

text( ...
    resultsAxes, ...
    0.5, ...
    0.125, ...
    ['Species outputs represent probable closed-set candidates and ' ...
     'not definitive botanical identification.'], ...
    'HorizontalAlignment','center', ...
    'FontSize',9.4, ...
    'Color',[0.35 0.35 0.35]);

exportgraphics( ...
    figResults, ...
    fullfile( ...
    folders.figures, ...
    'project_results.png'), ...
    'Resolution',300);

close(figResults);


%% ========================================================================
% 21. PROJECT SUMMARY
% ========================================================================

metric = [
    "Balanced vegetation dataset"
    "Vegetation training set"
    "Vegetation validation set"
    "Vegetation independent test set"
    "Vegetation classifier accuracy"
    "Closed-set species classes"
    "Closed-set species classifier accuracy"
    "Historical terrestrial images"
    "Historical observation points"
    "2010 mean NDVI"
    "2024 mean NDVI"
    "2024 dense vegetation"
    "2026 mean NDVI"
    "2026 dense vegetation"
    "NDVI change from 2010 to 2026"
];

result = [
    string(vegetationDatasetSize) + " images"
    string(vegetationTrainingSize) + " images"
    string(vegetationValidationSize) + " images"
    string(vegetationTestSize) + " images"
    sprintf('%.2f %%',vegetationAccuracy)
    "8 classes"
    sprintf('%.2f %%',speciesAccuracy)
    string(historicalImageCount) + " images"
    "9 points"
    sprintf('%.3f',meanNDVI(ndviYear == 2010))
    sprintf('%.3f',meanNDVI(ndviYear == 2024))
    sprintf('%.2f %%',denseVegetationPercent(ndviYear == 2024))
    sprintf('%.3f',meanNDVI(ndviYear == 2026))
    sprintf('%.2f %%',denseVegetationPercent(ndviYear == 2026))
    sprintf('%.1f %%',ndviChange2010to2026)
];

source = [
    vegetationResultSource
    vegetationResultSource
    vegetationResultSource
    vegetationResultSource
    vegetationResultSource
    speciesResultSource
    speciesResultSource
    historicalResultSource
    historicalResultSource
    "Archived validated satellite result"
    "Archived validated satellite result"
    "Archived validated satellite result"
    "Archived validated satellite result"
    "Archived validated satellite result"
    "Calculated from archived validated NDVI values"
];

projectSummary = table( ...
    metric, ...
    result, ...
    source, ...
    'VariableNames',{ ...
        'Metric', ...
        'Result', ...
        'Provenance'});

writetable( ...
    projectSummary, ...
    fullfile( ...
    folders.results, ...
    'project_summary.csv'));

fprintf('\nFinal project summary:\n');

disp(projectSummary);


%% ========================================================================
% 22. RUN METADATA
% ========================================================================

metadataFile = ...
    fullfile(folders.results,'run_metadata.txt');

fileID = fopen(metadataFile,'w');

if fileID ~= -1

    fprintf(fileID,'Remote Vegetation Characterization Project\n');
    fprintf(fileID,'Generated: %s\n\n',char(datetime('now')));

    fprintf(fileID,'Vegetation results source:\n%s\n\n', ...
        vegetationResultSource);

    fprintf(fileID,'Species results source:\n%s\n\n', ...
        speciesResultSource);

    fprintf(fileID,'Historical image source:\n%s\n\n', ...
        historicalResultSource);

    fprintf(fileID,[ ...
        'NDVI values are archived validated outputs from the original ' ...
        'satellite-processing stage.\n']);

    fclose(fileID);

end


%% ========================================================================
% 23. FINAL OUTPUT
% ========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('PIPELINE COMPLETED\n');
fprintf('============================================================\n');

fprintf('\nGenerated figures:\n');
fprintf('images/project_workflow.png\n');
fprintf('images/confusion_matrix.png\n');
fprintf('images/ndvi_analysis.png\n');
fprintf('images/project_results.png\n');

fprintf('\nGenerated result files:\n');
fprintf('results/data_sources.csv\n');
fprintf('results/vegetation_classifier_metrics.csv\n');
fprintf('results/vegetation_confusion_matrix.csv\n');
fprintf('results/species_classifier_metrics.csv\n');
fprintf('results/ndvi_time_series.csv\n');
fprintf('results/project_summary.csv\n');
fprintf('results/run_metadata.txt\n');

fprintf('\nKey methodological limitations:\n');
fprintf('- No on-site vegetation survey was available.\n');
fprintf('- Species identification is probabilistic and closed-set.\n');
fprintf('- Regional occurrence data do not prove local occurrence.\n');
fprintf('- Sentinel-2 cannot reliably identify individual tree crowns.\n');
fprintf('- Landsat and Sentinel-2 have different spatial resolutions.\n');

fprintf('\n============================================================\n');


%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function ensureFolder(folderPath)

    if ~isfolder(folderPath)
        mkdir(folderPath);
    end

end


function folderPath = firstExistingFolder(candidateFolders)

    folderPath = "";

    for i = 1:numel(candidateFolders)

        if isfolder(candidateFolders{i})

            folderPath = string(candidateFolders{i});
            return

        end

    end

end


function balancedData = balanceImageDatastore(imageData)

    labelSummary = countEachLabel(imageData);

    targetCount = min(labelSummary.Count);

    classNames = categories(imageData.Labels);

    selectedIndices = [];

    for i = 1:numel(classNames)

        currentIndices = find( ...
            imageData.Labels == classNames{i});

        randomSelection = randperm( ...
            numel(currentIndices), ...
            targetCount);

        selectedIndices = [
            selectedIndices
            currentIndices(randomSelection)
        ];

    end

    selectedIndices = ...
        selectedIndices(randperm(numel(selectedIndices)));

    balancedData = ...
        subset(imageData,selectedIndices);

end


function cleanData = removeExactDuplicateImages(imageData)

    numberOfFiles = numel(imageData.Files);

    fileHashes = strings(numberOfFiles,1);

    for i = 1:numberOfFiles

        try

            fileHashes(i) = ...
                calculateFileMD5(imageData.Files{i});

        catch

            % Preserve the file if hashing fails.
            fileHashes(i) = ...
                "UNHASHED_" + string(i);

        end

    end

    [~,uniqueIndices] = ...
        unique(fileHashes,'stable');

    cleanData = ...
        subset(imageData,uniqueIndices);

    removedCount = ...
        numberOfFiles-numel(uniqueIndices);

    fprintf('Exact duplicate files removed: %d\n', ...
        removedCount);

end


function hashString = calculateFileMD5(filePath)

    fileID = fopen(filePath,'rb');

    if fileID == -1
        error('Could not open file: %s',filePath);
    end

    cleanupObject = onCleanup(@() fclose(fileID));

    fileBytes = fread( ...
        fileID, ...
        Inf, ...
        '*uint8');

    messageDigest = ...
        java.security.MessageDigest.getInstance('MD5');

    messageDigest.update( ...
        typecast(fileBytes,'int8'));

    digestBytes = ...
        typecast(messageDigest.digest(),'uint8');

    hashString = ...
        lower(string(sprintf('%02x',digestBytes)));

    clear cleanupObject

end


function validateVegetationClasses(labels,expectedKeys)

    actualCategories = ...
        string(categories(labels));

    actualKeys = ...
        canonicalVegetationLabels(actualCategories);

    missingClasses = ...
        expectedKeys(~ismember(expectedKeys,actualKeys));

    if ~isempty(missingClasses)

        error( ...
            'Expected vegetation classes are missing: %s', ...
            strjoin(missingClasses,', '));

    end

end


function keys = canonicalVegetationLabels(labels)

    keys = lower(string(labels));

    keys = replace( ...
        keys, ...
        ["á","à","ã","â","é","ê","í","ó","ô","õ","ú","ç"], ...
        ["a","a","a","a","e","e","i","o","o","o","u","c"]);

    keys = regexprep(keys,'[\s\-]+','_');

    for i = 1:numel(keys)

        switch keys(i)

            case {"tree","arvore"}
                keys(i) = "arvore";

            case {"shrub","arbusto"}
                keys(i) = "arbusto";

            case {"bare_soil","soil","solo_exposto"}
                keys(i) = "solo_exposto";

            case { ...
                    "ground_vegetation", ...
                    "low_vegetation", ...
                    "vegetacao_rasteira"}

                keys(i) = "vegetacao_rasteira";

        end

    end

end


function confusionMatrix = ...
    buildVegetationConfusionMatrix( ...
        trueLabels, ...
        predictedLabels, ...
        classOrder)

    trueKeys = ...
        canonicalVegetationLabels(trueLabels);

    predictedKeys = ...
        canonicalVegetationLabels(predictedLabels);

    unknownTrue = ...
        ~ismember(trueKeys,classOrder);

    unknownPredicted = ...
        ~ismember(predictedKeys,classOrder);

    if any(unknownTrue)

        error( ...
            'Unknown true vegetation labels were found: %s', ...
            strjoin(unique(trueKeys(unknownTrue)),', '));

    end

    if any(unknownPredicted)

        error( ...
            'Unknown predicted vegetation labels were found: %s', ...
            strjoin(unique(predictedKeys(unknownPredicted)),', '));

    end

    numberOfClasses = ...
        numel(classOrder);

    confusionMatrix = ...
        zeros(numberOfClasses);

    for row = 1:numberOfClasses

        for column = 1:numberOfClasses

            confusionMatrix(row,column) = sum( ...
                trueKeys == classOrder(row) & ...
                predictedKeys == classOrder(column));

        end

    end

end


function confusionMatrix = ...
    buildGenericConfusionMatrix( ...
        trueLabels, ...
        predictedLabels, ...
        classNames)

    trueStrings = string(trueLabels);
    predictedStrings = string(predictedLabels);
    classNames = string(classNames);

    numberOfClasses = numel(classNames);

    confusionMatrix = zeros(numberOfClasses);

    for row = 1:numberOfClasses

        for column = 1:numberOfClasses

            confusionMatrix(row,column) = sum( ...
                trueStrings == classNames(row) & ...
                predictedStrings == classNames(column));

        end

    end

end


function metricsTable = ...
    metricsFromConfusionMatrix( ...
        confusionMatrix, ...
        classNames)

    truePositive = ...
        diag(confusionMatrix);

    predictedTotals = ...
        sum(confusionMatrix,1)';

    actualTotals = ...
        sum(confusionMatrix,2);

    precision = ...
        truePositive ./ predictedTotals;

    recall = ...
        truePositive ./ actualTotals;

    f1 = ...
        2 .* ...
        (precision .* recall) ./ ...
        (precision + recall);

    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    f1(isnan(f1)) = 0;

    support = actualTotals;

    metricsTable = table( ...
        string(classNames(:)), ...
        support, ...
        precision*100, ...
        recall*100, ...
        f1*100, ...
        'VariableNames',{ ...
            'Class', ...
            'Support', ...
            'Precision_percent', ...
            'Recall_percent', ...
            'F1_percent'});

end


function historicalTable = ...
    buildHistoricalImageInventory(historicalFolder)

    imageFiles = [
        dir(fullfile(historicalFolder,'**','*.png'))
        dir(fullfile(historicalFolder,'**','*.jpg'))
        dir(fullfile(historicalFolder,'**','*.jpeg'))
    ];

    numberOfFiles = ...
        numel(imageFiles);

    pointNumber = ...
        nan(numberOfFiles,1);

    year = ...
        nan(numberOfFiles,1);

    filePath = ...
        strings(numberOfFiles,1);

    validYears = ...
        [2010 2011 2018 2019 2022 2024];

    for i = 1:numberOfFiles

        currentFolder = ...
            imageFiles(i).folder;

        currentName = ...
            imageFiles(i).name;

        filePath(i) = string( ...
            fullfile( ...
                currentFolder, ...
                currentName));

        pointToken = regexp( ...
            currentFolder, ...
            'PONTO\s*(\d+)', ...
            'tokens', ...
            'once', ...
            'ignorecase');

        if ~isempty(pointToken)

            pointNumber(i) = ...
                str2double(pointToken{1});

        end

        yearToken = regexp( ...
            currentName, ...
            '(2010|2011|2018|2019|2022|2024)', ...
            'match', ...
            'once');

        if ~isempty(yearToken)

            year(i) = ...
                str2double(yearToken);

        end

    end

    validRecord = ...
        ~isnan(pointNumber) & ...
        ~isnan(year) & ...
        pointNumber >= 0 & ...
        pointNumber <= 8 & ...
        ismember(year,validYears);

    pointNumber = ...
        pointNumber(validRecord);

    year = ...
        year(validRecord);

    filePath = ...
        filePath(validRecord);

    historicalTable = table( ...
        pointNumber, ...
        year, ...
        filePath, ...
        'VariableNames',{ ...
            'ObservationPoint', ...
            'Year', ...
            'File'});

    historicalTable = ...
        sortrows( ...
            historicalTable, ...
            {'ObservationPoint','Year'});

end


function speciesTable = ...
    fetchINaturalistPlants( ...
        latitude, ...
        longitude, ...
        radiusKm)

    endpoint = ...
        'https://api.inaturalist.org/v1/observations/species_counts';

    options = weboptions( ...
        'Timeout',30, ...
        'ContentType','json');

    response = webread( ...
        endpoint, ...
        'lat',latitude, ...
        'lng',longitude, ...
        'radius',radiusKm, ...
        'quality_grade','research', ...
        'iconic_taxa','Plantae', ...
        'hrank','species', ...
        'per_page',200, ...
        options);

    numberOfResults = ...
        numel(response.results);

    scientificName = ...
        strings(numberOfResults,1);

    observationCount = ...
        zeros(numberOfResults,1);

    for i = 1:numberOfResults

        currentResult = ...
            response.results(i);

        observationCount(i) = ...
            currentResult.count;

        if isfield(currentResult,'taxon') && ...
           isfield(currentResult.taxon,'name')

            scientificName(i) = ...
                string(currentResult.taxon.name);

        else

            scientificName(i) = ...
                "Unknown";

        end

    end

    speciesTable = table( ...
        scientificName, ...
        observationCount, ...
        'VariableNames',{ ...
            'ScientificName', ...
            'NearbyObservations'});

    speciesTable = sortrows( ...
        speciesTable, ...
        'NearbyObservations', ...
        'descend');

end