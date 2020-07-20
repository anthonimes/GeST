function [SegCentre,RImageCentreY,RImageCentreX] = SegmentCentre(MergeImage) % Segment centre of region image

[Y,X,~] = size(MergeImage);

% Find the centre point of MergeImage
ImageCentreX = round(X/2);
ImageCentreY = round(Y/2);

% Find the segment number of MergeImage middle point
MergeSegNumber = MergeImage(ImageCentreY,ImageCentreX);

% Initialize the 4 variables which store the X or Y edge value of the segment.
% (Leftmost, Rightmost, Upper most and Bottom most)
LeftX = ImageCentreX;
RightX = ImageCentreX;
UpY = ImageCentreY;
BottomY = ImageCentreY;

% Find the value (4 variables)
for j = 1:1:Y
    for i = 1:1:X
        if MergeImage(j,i) == MergeSegNumber 
            if i < LeftX
                LeftX = i;
            end
            
            if i > RightX
                RightX = i;
            end
            
            if j < UpY
                UpY = j;
            end
            
            if j > BottomY
                BottomY = j;
            end      
        end
    end
end

% Find the centre point of RegionImage
RImageCentreX = uint16(RightX+LeftX)/2;
RImageCentreY = uint16(BottomY+UpY)/2;

% SegCentre = RegionImage(RImageCentreY,RImageCentreX);
SegCentre = MergeImage(RImageCentreY,RImageCentreX);

end
