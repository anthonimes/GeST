function RegionImage = SegmentsOfInterest(RegionImage,SegNumArray)

[m,n,~] = size(RegionImage);
[~,y] = size(SegNumArray);

% Let the region we want = 0
for k = 1:1:y  %% To be tested
    SegNum = SegNumArray(1,k);
    for j = 1:1:m
        for i = 1:1:n
            if RegionImage(j,i) == SegNum
                RegionImage(j,i) = 0;
            end
        end
    end
    
end

end