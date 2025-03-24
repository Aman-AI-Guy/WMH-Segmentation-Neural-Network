% Load the image in grayscale
gray_image = imread('brain.jpg');
gray_image = double(gray_image);
gray_image = gray_image ./ max(gray_image(:));

wmh_image = imread('mask.jpg');
wmh_image = double(wmh_image);
wmh_image = wmh_image ./ max(wmh_image(:));


dimY = size(gray_image,1);
dimX = size(gray_image,2);



%% Initialize a matrix for the WMH image
imgwmh = zeros(dimY, dimX);


% Read the FIS (Fuzzy Inference System) file for the wmhsegmentation
 wmhsegmentation= readfis('WmhSegmentation.fis');

% Calculate the features of images using a 3x3 sliding window
% Compute local statistics using colfilt
media = colfilt(gray_image, [3 3], 'sliding', @mean);
devStandard = colfilt(gray_image, [3 3], 'sliding', @std);
kurt = colfilt(gray_image, [3 3], 'sliding', @kurtosis);
skew = colfilt(gray_image, [3 3], 'sliding', @skewness);
kurt(isnan(kurt)) = 0;
skew(isnan(skew)) = 0;

media = normalize_feature(media);
devStandard = normalize_feature(devStandard);
kurt = normalize_feature(kurt);
skew = normalize_feature(skew);

% Scan the image pixel by pixel
for i = 1 : dimY
    for j = 1 : dimX
        % Evaluate the FIS for each pixel using the features
        imgwmh(i,j) = evalfis(wmhsegmentation, [gray_image(i,j), media(i,j), devStandard(i,j), kurt(i,j), skew(i,j)]);
    end
end
%%
% Display the WMH (White Matter Hyperintensity) image
figure
imshow(imgwmh,[])
title('WMH')

% Function to normalize feature to range [0, 1]
function normalized_data = normalize_feature(data)
    min_val = min(data(:));
    max_val = max(data(:));
    normalized_data = (data - min_val) / (max_val - min_val);
end


