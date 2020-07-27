%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PART3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% The fundamental matrix of the stereo system%%%%%%%%%%%%%%%
%% FUNDAMENTAL MATRIX

I1 = imread('left.png');
I2 = imread('right.png');

%load the matched points 
load stereoPointPairs
disp(matched_points1)
disp(matched_points2)

%plotting the matched points
figure;
showMatchedFeatures(I1,I2,matched_points1,matched_points2,'montage','PlotOptions',{'ro','go','y--'});

title('Putative point matches');


%Use the Least Median of Squares Method to Find Inliers
[fLMedS, inliers] = estimateFundamentalMatrix(matched_points1,matched_points2,'NumTrials',2000);

%Show the inliers points
figure;
showMatchedFeatures(I1, I2, matched_points1(inliers,:),matched_points2(inliers,:),'montage','PlotOptions',{'ro','go','y--'});
title('Point matches after outliers were removed');

%compute the fundamental matrix
inlierPts2 = matched_points2(knownInliers,:);
inlierPts1 = matched_points1(knownInliers,:);
fNorm8Point = estimateFundamentalMatrix(inlierPts1,inlierPts2,'Method','Norm8Point');
disp(fNorm8Point);

%% DISPARITY MAP

%%%%%%%%%%% The disparity map %%%%%%%%%%%%%%
%%Show stereo anaglyph. Use red-cyan stereo glasses to view image in 3-D.

%figure
%imshow(stereoAnaglyph(I1,I2));
%title('Red-cyan composite view of the stereo images');

disparityRange = [-6,10];
disparityMap = disparity(rgb2gray(I1),rgb2gray(I2),'BlockSize',15,'DisparityRange',disparityRange);

%Compute the disparity map.
figure 
imshow(disparityMap,disparityRange);
title('Disparity Map');
colormap(gca,jet) 
colorbar




