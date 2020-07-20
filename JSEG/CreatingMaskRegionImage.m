function Mask = CreatingMaskRegionImage(RegionImage)

[m,n,d] = size(RegionImage);

for j = 1:1:m
    for i = 1:1:n
        if RegionImage(j,i) == 0
            Mask(j,i) = 1;
        else
            Mask(j,i) = 0;
        end
    end
end

end
