%STEC: Sampling-based Two-layer Ensemble Clustering

clear;clc
close all
addpath('functions')

load SF2M.mat
warning('off')
X = fea;
Y_true = gt;

numRSP_sample = 100;
repeatTime = 20;
ticStart = tic;
for iter_exp = 1:repeatTime
    % Generate random samples of big data
    [num,dim] = size(X);
    indexanchor_sample = randi([1, num], [1, num]);
    interval = floor(num/numRSP_sample);
    for iter = 1:numRSP_sample
            if iter~=numRSP_sample
                X_sample_RSP(:,:,iter)= X(indexanchor_sample((iter-1)*interval+1:iter*interval),:);
            else
                X_sample_RSP(:,:,iter)= X(indexanchor_sample(end-interval+1:end),:);
            end 
    end
    %Generate base clustering results of global random sample
    k = numel(unique(Y_true));  
    E = 20; %Number of base clusterings
    t = 15; %Ensemble size (Second layer)
    upK = k*16;
    lowK = k*15;
    Ks = randsample(upK-lowK+1,E,true)+lowK-1; %Number of clusters in first layer
    Ks_new_upK = k*3;
    Ks_new_lowK = k*3;
    Ks_new = randsample(Ks_new_upK-Ks_new_lowK+1,E,true)+Ks_new_lowK-1; %Number of clusters in second layer
    C_model = [];
    if numRSP_sample > E
        numRSP_sample_select = randperm(numRSP_sample-1,E+1);
    else
        numRSP_sample_select = randi(numRSP_sample-1,1,E+1);
    end
    for iter = 1:E
        [y_temp,C_temp]=kmeans(X_sample_RSP(:,:,numRSP_sample_select(iter)),Ks(iter),'MaxIter',30);
        C = C_temp;
        C_model = [C_model;C];
        Y_C = 1:Ks(iter);
        savefile = ['tempdata/C',num2str(iter),'.mat'];
        save(savefile,'C','Y_C')      
    end
    X_sample = X_sample_RSP(:,:,numRSP_sample_select(E+1));
    %Generate B matrix
    Y_all = [];
    for iter = 1:E
        B = [];
        loadfile = ['tempdata/C',num2str(iter),'.mat'];
        load(loadfile,'C','Y_C')
        Mdl = fitcknn(C,Y_C);
        Y_sample = predict(Mdl,X_sample);
        Y_all = [Y_all,Y_sample];
        B = generateBinaryMatrix(Y_sample,Ks(iter));
        savefile = ['tempdata/B',num2str(iter),'.mat'];
        save(savefile,'B')
        clear B
    end
   t0 = 8;

    %ensemble clustering
    maxKmIters=50;
    Y_all_new = [];
    B4 = [];   
    for iter2 = 1:t    
        [Y_meta,KK] = calculateEnsembleFinal(Y_all,E,t0,Ks_new(iter2));
        B4 = [B4,generateBinaryMatrix(Y_meta,Ks_new(iter2))];
    end
    B4_old = B4;
    % BSGP
    B4 = B4./sqrt(t)*diag(1./(sqrt(sum(B4)+1e-6))); 
    B4TB4 = B4'*B4;
    [V1, D1] = eigs(B4TB4,k);
    V1 = bsxfun( @rdivide, V1, sqrt(sum(V1.*V1,2)) + 1e-10 );
    Y_cluster_final = kmeans(V1,k,'MaxIter',maxKmIters);
    %PVE*
    U = generateBinaryU(Y_cluster_final,k);
    Y_sample_temp = B4_old*U';
    [~,Y_sample_ensemble_clustering] = max(Y_sample_temp,[],2);

    %Generating clustering result of big data
    Mdl_sample = fitcknn(X_sample,Y_sample_ensemble_clustering);
    Y_C_all = predict(Mdl_sample,C_model);   
    Mdl_C_all = fitcknn(C_model,Y_C_all);
    Y_clustering = predict(Mdl_C_all,X);
    clear B3
    result(iter_exp,:) = ClusteringMeasure_new(Y_clustering,Y_true) 
    clear Y_clustering
end

runningTime = toc(ticStart);
ave = mean(result)
std = std(result)
time = runningTime/20
save result_temp.mat result time

function [Y_sample,KK] = calculateEnsembleFinal(Y_all_input,E,F,Ke_new)
    KK = randperm(E);
    B3 = [];
    for iter1 = 1:F
        loadfile = ['tempdata/B',num2str(KK(iter1)),'.mat'];
        load(loadfile,'B')
        B3 = [B3,B];
    end
    Y_all = Y_all_input(:,KK(1:F));
    k = Ke_new;
    maxKmIters=50;
    B3_old = B3;
    B3 = B3./sqrt(F)*diag(1./sqrt(sum(B3)));  
    B3TB3 = B3'*B3;
    [V1, ~] = eigs(B3TB3,k);
    V1 = bsxfun( @rdivide, V1, sqrt(sum(V1.*V1,2)) + 1e-10 );
    Y_cluster = kmeans(V1,k,'MaxIter',maxKmIters);
    U = generateBinaryU(Y_cluster,k);
    Y_sample_temp = B3_old*U';
    [~,Y_sample] = max(Y_sample_temp,[],2);
end

function U = generateBinaryU(Y_cluster,c)
  [num,~]=size(Y_cluster);
  U = zeros(c,num);
  for i=1:c
      index = find(Y_cluster==i);
      U(i,index) = 1;
      clear index
  end
end

function BY = generateBinaryMatrix(y,c)
  [num,~]=size(y);
  BY = zeros(num,c);
  for i = 1:c
      index = find(y==i);
      BY(index,i)=1;
  end
  BY = sparse(BY);
end
