% Load grayscale and WMH_mask image and perform filtering operations
gray_image = imread('brain.jpg');
gray_image = double(gray_image);
gray_image = normalize_feature(gray_image);

wmh_image = imread('mask.jpg');
wmh_image = double(wmh_image);
wmh_image = normalize_feature(wmh_image);


% Compute local statistics using colfilt
media = colfilt(gray_image, [3 3], 'sliding', @mean);
devStandard = colfilt(gray_image, [3 3], 'sliding', @std);
kurt = colfilt(gray_image, [3 3], 'sliding', @kurtosis);
skew = colfilt(gray_image, [3 3], 'sliding', @skewness);
kurt(isnan(kurt)) = 0;
skew(isnan(skew)) = 0;

% Compute local statistics using colfilt
g_media = colfilt(wmh_image, [3 3], 'sliding', @mean);
g_devStandard = colfilt(wmh_image, [3 3], 'sliding', @std);
g_kurt = colfilt(wmh_image, [3 3], 'sliding', @kurtosis);
g_skew = colfilt(wmh_image, [3 3], 'sliding', @skewness);
g_kurt(isnan(g_kurt)) = 0;
g_skew(isnan(g_skew)) = 0;

% Normalizing the featues
media = normalize_feature(media);
devStandard = normalize_feature(devStandard);
kurt = normalize_feature(kurt);
skew = normalize_feature(skew);

g_media = normalize_feature(g_media);
g_devStandard = normalize_feature(g_devStandard);
g_kurt = normalize_feature(g_kurt);
g_skew = normalize_feature(g_skew);


gold_standard.Inputs = [gray_image(:), media(:), devStandard(:), kurt(:), skew(:)];
gold_standard.Outputs = [wmh_image(:), g_media(:), g_devStandard(:), g_kurt(:), g_skew(:)];


assert(size(gold_standard.Inputs, 2) == 5);

% Load the FIS file
fis = readfis('WmhSegmentation.fis');

% Defining the fitness function
FitnessFunction = @(weights) evaluate_fis_fitness(fis, gold_standard);

% Defining the number of variables 
numberOfVariables = length(fis.Rules);

% Defining lower and upper bounds for the weights
lb = zeros(1, numberOfVariables); 
ub = ones(1, numberOfVariables);  

% Set up options for the genetic algorithm
opts = optimoptions('ga', ...
    'PopulationType', 'doubleVector', ...
    'SelectionFcn', @selectionroulette, ...
    'CrossoverFcn', @crossoversinglepoint, ...
    'CrossoverFraction', 0.8, ...
    'PopulationSize', 20, ...
    'MutationFcn', @mutationadaptfeasible, ... 
    'MaxGenerations', 1, ...
    'EliteCount', 1);

% Run the genetic algorithm
[x, fval, exitflag, output, final_pop, scores] = ga(FitnessFunction, numberOfVariables, [], [], [], [], lb, ub, [], opts);

% Display the best solution found
disp('Best Weights:');
disp(x);


%% Function definitions

% Function to evaluate the fitness of an individual
function fitness = evaluate_fis_fitness(fis, gold_standard)
   
    % Evaluate the FIS with the given inputs
    prediction = evalfis(fis, gold_standard.Inputs); % FIS prediction
    
    % Flatten the prediction to match the output format
    prediction = prediction(:);
    
    % Calculate the fitness (inverse of error)
    error = sum((prediction - gold_standard.Outputs).^2, 'all');
    fitness = 1 / (1 + error);
end

% Function to normalize feature to range [0, 1]
function normalized_data = normalize_feature(data)
    min_val = min(data(:));
    max_val = max(data(:));
    normalized_data = (data - min_val) / (max_val - min_val);
end

