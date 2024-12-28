function [F1, AUC, MCC] = ALOGO(source, target)
%ALOGO Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) source - a n1*(d+1) matrix, the last column is the label where 0/1
%   denotes the non-defective/defective module.
%   (2) target - a n2*(d+1) matrix, the last column is the label where 0/1
%   denotes the non-defective/defective module.
%   (3) LOC - Line-of-code metric.
%   (4) order - incoming order of target instances.  
% OUTPUTS:
%   Performance measures.


rand('seed',0);

% Train a offline CPDP model
[PD,PF,Precision,F1,AUC,Accuracy,G_measure,MCC,Balance, probPosCPDP] = ARRAY(source, target, LOC);

twoClassFlag = false;
probPos = zeros(size(target, 1),1); % 
probPos(1) = probPosCPDP(1);
probPosWPDP = zeros(size(target, 1),1);
probPosWPDP(1) = 0;

% The label of target data arrives in streaming manner
for i=2:size(target, 1) % each target instance except the 1st traget instance
    labeledTarget = target(1:i-1,:);
    if ~twoClassFlag
        if numel(unique(labeledTarget(:,end)))==2
            twoClassFlag = true;
        end
        probPos(i) = probPosCPDP(i); 
    else
        % 
        if sum(labeledTarget(:,end)==1)>5
            labeledTarget = SMOTE_02(labeledTarget,1); 
        end
        
        
        % train a online WPDP classifier
        nTrees = 50;
        trainX = labeledTarget(:,1:end-1);
        trainY = labeledTarget(:,end);
        RF = TreeBagger(nTrees, trainX, trainY, 'Method','classification');
        % Prediction
        [~, score] = predict(RF, target(i,1:end-1));
        probPosWPDP(i) = score(:,end);
        
        % Adaptive weight adjustment
        if i>ceil(size(target,1)/4)
            probPos(i) = probPosWPDP(i);
        else
            lamda = i/size(target, 1);
            probPos(i) = (1-lamda)*probPosCPDP(i) + lamda*probPosWPDP(i);
        end
    end
end

try
    [ PD,PF,Precision,F1,AUC,Accuracy,G_measure,MCC,Balance] = Performance(target(:,end), probPos); 
catch
    PD=nan;PF=nan;Precision=nan;F1=nan;AUC=nan;Accuracy=nan;G_measure=nan;MCC=nan;Balance=nan; probPos = nan;
end

end

