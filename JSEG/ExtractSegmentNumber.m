function TEST3 = ExtractSegmentNumber(RegionImage)

[m,n,d] = size(RegionImage);

% Find surrounding segments of SegCentre
for j = 1:1:m
    for i = 1:1:n
        if i == n || i == 1 || j == 1 || j == m
            TEST1(j,i) = 0;
        else
            if RegionImage(j,i) == 0
                TEST1(j,i) = 0;
            else
                A = RegionImage(j,i)*RegionImage(j,i+1)*RegionImage(j,i-1)*RegionImage(j+1,i)*RegionImage(j-1,i);
                if A == 0
                    TEST1(j,i) = RegionImage(j,i);
                else
                    TEST1(j,i) = 0;
                end
            end
        end
    end
end

% To extract segment numbers that surrounded centre segment
k = 0;

for j = 1:1:m
    for i = 1:1:n
        if TEST1(j,i) == 0
        else
            k = k + 1;
            TEST2(k) = TEST1(j,i);
        end
    end
end


for i = 1:1:k
    A = TEST2(i);
    
    for j = 1:1:k
        if i == j
        else
            if A == TEST2(j)
                TEST2(j) = 0;
            end
        end
    end
    
end

l = 0;

for i = 1:1:k-1
    if TEST2(i) == 0
    else
        l = l + 1;
        TEST3(l) = TEST2(i);
    end
end

end

