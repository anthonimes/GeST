%% Pre-processing: Creating mask

function BWdfill = PreprocessMask(I)
%%Image Resize
%I = imresize(I,0.11);

%% Median filering
Filter1 = medfilt2(I(:,:,1));
Filter2 = medfilt2(I(:,:,2));
Filter3 = medfilt2(I(:,:,3));

I(:,:,1) = Filter1;
I(:,:,2) = Filter2;
I(:,:,3) = Filter3;

%% Convert image from RGB to HSV model
HSV_Image = rgb2hsv(I);

%% Binarized image
%level = graythresh(HSV_Image(:,:,2));
HSV_Image3 = im2bw(HSV_Image(:,:,2),0.1);
%figure, imshow(HSV_Image3), title('Grey_Image');

%% Closing an image (Dilation and then erosion)
se = strel('disk',10);
I_dil = imdilate(HSV_Image3, se);
erodedBW = imerode(I_dil,se);
%figure, imshow(erodedBW), title('dilated gradient mask');

%% Filling interior gape
BWdfill = imfill(erodedBW,8,'holes');
%figure, imshow(BWdfill),title('binary image with filled holes');

%% Clear Boader
BWdfill = imclearborder(BWdfill);

end
