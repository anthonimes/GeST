function TEST3 = ExtractMinorSegmentNumber(tmpImage)

[m,n,d] = size(tmpImage);

% To extract segment numbers that surrounded centre segment
k = 0;

for j = 1:1:m
    for i = 1:1:n
        if tmpImage(j,i) == 0
        else
            k = k + 1;
            TEST2(k) = tmpImage(j,i);
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

