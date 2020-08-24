addpath lib;

clear all;close all;clc;

imgDir = '/home/anthony/LIFO/jouets/segmentation/dataset/XP/BSR/val/';
outDir = '/home/anthony/LIFO/jouets/segmentation/dataset/XP/BSR/val_labels/';
mkdir(outDir);
D= dir(fullfile(imgDir,'*.mat'));

tic;
for i =1:numel(D),
    nthresh = 99;
    thresh = linspace(1/(nthresh+1),1-1/(nthresh+1),nthresh)';
    disp( thresh )
    outFile = fullfile(outDir,[D(i).name(1:end-4) '.mat']);
    
    umcFile=strcat(imgDir,D(i).name)
    load(umcFile,'ucm2');
    ucm = double(ucm2);

    % get superpixels at scale k without boundaries:
    labels2 = bwlabel(ucm <= 0.166);
    seg = labels2(2:2:end, 2:2:end);

    save(outFile, 'seg');
end
toc;