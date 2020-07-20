function EdgeMask = RemoveEdge (Image)

%% Remove edge
[m,n] = size(Image);

for j = 1:1:m
    for i = 1:1:n
        if (i == 1) || (i == n) || (j == 1) || (j == m)
            tmpImage(j,i) = Image(j,i);
        else
            tmpImage(j,i) = 0;
        end
    end
end

SegNumArray = ExtractMinorSegmentNumber(tmpImage);

Image = SegmentsOfInterest(Image,SegNumArray);

for j = 1:1:m
    for i = 1:1:n
        if Image(j,i) == 0
            EdgeMask(j,i) = 0;
        else
            EdgeMask(j,i) = 1;
        end
    end
end

end