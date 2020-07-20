function [Erout,Mout, nbout, Pout] = kmeansO(Xin,kmax)
% kmeans - clustering with k-means (or Generalized Lloyd or LBG) algorithm
%
% [Er,M,nb] = kmeans(X,T,kmax,dyn,dnb,killing,p)
%
% X    - (n x d) d-dimensional input data
% T    - (? x d) d-dimensional test data
% kmax - (maximal) number of means
% dyn  - 0: standard k-means, unif. random subset of data init.
%        1: fast global k-means
%        2: non-greedy, just use kdtree to initiallize the means
%        3: fast global k-means, use kdtree for potential insertion locations
%        4: global k-means algorithm
% dnb  - desired number of buckets on the kd-tree
% pl   - plot the fitting process
%
% returns
% Er - sum of squared distances to nearest mean (second column for test data)
% M  - (k x d) matrix of cluster centers; k is computed dynamically
% nb - number of nodes on the kd-tree (option dyn=[2,3])
% P  - Partition labels
%
% Nikos Vlassis & Sjaak Verbeek, 2001, http://www.science.uva.nl/~jverbeek
% modify by Qinpei, 2009

X =gpuArray(Xin);
%kmax = gpuArray(kmaxin);


Er=[]; %TEr=[];              % error monitorring

[n,~]     = size(X);
P = zeros(n,1);

THRESHOLD = 1e-4;   % relative change in error that is regarded as convergence
nb        = 0;


k      = kmax;
tmp    = randperm(n);
M      = X(tmp(1:k),:);


Wold = realmax;

while k <= kmax
    kill = [];
    
    % squared Euclidean distances to means; Dist (k x n)
    Dist = sqdist(M',X');
    
    % Voronoi partitioning
    [Dwin,Iwin] = min(Dist',[],2);
    P = Iwin;
    
    % error measures and mean updates
    Wnew = sum(Dwin);
    
    % update VQ's
    for i=1:size(M,1)
        I = Iwin==i;
        M(i,:) = mean(X(I,:));
    end
    
    if 1-Wnew/Wold < THRESHOLD*(10-9*(k==kmax))
%         if dyn && k < kmax
%         else
            k = kmax+1;
%         end
    end
    Wold = Wnew;
    
end

Er=[Er; Wnew];
%if ~isempty(T); tmp=sqdist(T',M'); TEr=[TEr; sum(min(tmp,[],2))]; Er=[Er TEr];end;
M(kill,:)=[];

Erout = gather(Er);
Mout = gather(M);
nbout = gather(nb);
Pout = gather(P);


