function [FinalImage] = JSEG(image_org)

%Load the image
%image_org = imread('Testing.JPG');

%color map number
colornum = 20;

%Scale Range
boxnum = 1024;

% Resize the image
image_org = imresize(image_org,0.1);

%====== partition ========%
% class_map from clustering algorithms
[m,n,d] = size(image_org);
% change all image values into 1 column and RGB(3 layers)
X = reshape(double(image_org), m*n,d);
%P = arrayfun(@kmeansO,gpuArray(X));
[~,~,~,P] = kmeansO(X,colornum);
%toc;

map = reshape(P, m, n);

%%
w1 = 1;
%w2 = 2;
%w3 = 3;
w4 = 4;
W1 = GenerateWindow(w1);
% W2 = GenerateWindow(w2);
% W3 = GenerateWindow(w3);
W4 = GenerateWindow(w4);

JI1 = JImage(map, W1);
%toc;
% JI2 = JImage(map, W2);
% toc;
% JI3 = JImage(map, W3);
% toc;
JI4 = JImage(map, W4);
%toc;

ImgQ = uint8(class2Img(map, image_org));


%%
%Region = zeros(m, n);

% --------------------scale 4--------------------
% scale 4
u = mean(JI4(:));
s = std(JI4(:));
Region = ValleyD(JI4, boxnum, u, s); % 4.1 Valley Determination
Region = ValleyG1(JI4, Region);  % 4.2.2 Growing
% Region = ValleyG1(JI3, Region);  % 4.2.3 Growing at next smaller scale
% Region = ValleyG1(JI2, Region);  % 4.2.3 Growing at next smaller scale
Region = ValleyG2(JI1, Region);  % 4.2.4 remaining pixels at the smallest scale
Region4 = Region;

 figure; imshow(ImgQ);
 hold on;
 DrawLine(Region);
 hold off;

% w = 1;
%     Region = SpatialSeg(JI1, Region, w);
%     Region = ValleyG2(JI1, Region);
%     Region1 = Region;
 
% Prepare EdgeMask that remove the neighbouring segments of the edge of
% picture.
[SegCentre,~,~] = SegmentCentre(Region4);
RegionImage = SegmentsOfInterest(Region4,SegCentre);
EdgeMask = RemoveEdge(Region4);
RegionImageMask = CreatingMaskRegionImage(RegionImage);

% Convert datatype of original image to double
image_org = double(image_org);

I = RegionImageMask.*EdgeMask;
%I = EdgeMask;
R = image_org(:,:,1);
G = image_org(:,:,2);
B = image_org(:,:,3);

r = uint8(R.*I);
g = uint8(G.*I);
b = uint8(B.*I);

FinalImage(:,:,1) = r;
FinalImage(:,:,2) = g;
FinalImage(:,:,3) = b;

FinalImage = uint8(FinalImage);

figure,imshow(FinalImage);
end