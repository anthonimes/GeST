function J = JCalculation(class_map, St)
% JSEG for color image segmentation implementation
% Calculate a J value for a class map from clustering results (whole image)
% Qinpei 
% input: class labels from clustering algorithms
%        M: centroid of the whole class map. for a certain window, it will
%        not change each time
% output: J value -- higher value indicates classess separated
%                 -- lower value indicates compactness

% NOTE: make sure the class_map starts from 1.

Sw = 0;
a = min(class_map(:));
b = max(class_map(:));
for Q = a: 1: b
    [x,y] = find(class_map == Q);
    % mean vector of the vectors with class label l
    if isempty(x) 
        continue;
    end
    m_l = [mean(x), mean(y)];
%     m_l = [median(x), median(y)];
    m1 = repmat(m_l, length(x), 1);    
    xy = [x, y];
    Dist1 = trace(sqdist(xy', m1'));   
    %Dist1 = trace(pdist2(xy, m1,'squaredeuclidean'));   
    Sw = Sw + Dist1;
end

J = (St-Sw)/Sw;

        
