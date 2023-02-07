%<||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||>%
%<||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||>%
%<||||||||||||||EXPERIMENTAL DESING TO GENERATING ARX MODELS WITH ALL POSSIBLE VARIABLE COMBINATIONS OF INPUTS||||||||||||||>%
%<||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||>%
%<||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||>%
%
%°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°%
%°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°%
%% Number of neurons and delays °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°%
%°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°%
%°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°%
%
% Setting up the NARX Neural Network
trainFcn        = 'trainlm';
inputDelays     = 1:13;
feedbackDelays  = 1:13;
hiddenLayerSize = 10;
NoTraining = 15;
j = 0;
% Variables = ["PW" "SI" "WD" "H" "HI" "P" "DP" "T" "WD"];
% Variables = ["PW" "SI" "WD" "H"];
%% Prepare inputs and targent
%
target = CleanData(1:3250,1);
Input  = CleanData(1:3250,:);
n = size(Input,2);
%

%
%
results = [];
while j < n - 1
    
    limit = n - j;  
    c     = nchoosek(1:n,limit);
%     cvCombinaitons = nchoosek(Variables,limit);
    j = j + 1;
    ac{j} = c;
%     acvComb{j} = cvCombinaitons;
    % Setting up the variables
    InputArray  = cell(size(c,1),1);
    timeArray   = zeros(size(c,1),1);
    netArray    = cell(size(c,1),1);
    performance = zeros(size(c,1),1);
    trainPerformance = zeros(size(c,1),1);
    valPerformance  = zeros(size(c,1),1);
    testPerformance = zeros(size(c,1),1);
    
    h = waitbar(0,'Iterating...','Name','Calculation process');
    
    for i = 1:size(c,1) 
        
        InputArray{i} = Input(:,c(i,:));
        T = tonndata(target,false,false);
        X = tonndata(InputArray{i},false,false);        
        net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
        netArray1    = cell(NoTraining,1);
        performance1 = zeros(NoTraining,1);
        trainPerformance1 = zeros(NoTraining,1);
        valPerformance1  = zeros(NoTraining,1);
        testPerformance1 = zeros(NoTraining,1);
        timeArray1 = zeros(NoTraining,1);
        
        for m = 1:NoTraining
            tic
            net.layers{1}.transferFcn = 'logsig'; % help nntransfer
            net.layers{2}.transferFcn = 'purelin';
            net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
            net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

            [x,xi,ai,t] = preparets(net,X,{},T);
            net.divideFcn  = 'dividerand';
            net.divideMode = 'time';
            net.divideParam.trainRatio = 70/100;
            net.divideParam.valRatio   = 15/100;
            net.divideParam.testRatio  = 15/100;
            net.performFcn             = 'mse';

            [net, tr] = train(net,x,t,xi,ai);            
            
            y = net(x,xi,ai);
            e = gsubtract(t,y);
            performance1(m) = perform(net,t,y);

            trainTargets = gmultiply(t,tr.trainMask);
            valTargets   = gmultiply(t,tr.valMask);
            testTargets  = gmultiply(t,tr.testMask);
            trainPerformance1(m) = perform(net,trainTargets,y);
            valPerformance1(m)  = perform(net,valTargets,y);
            testPerformance1(m) = perform(net,testTargets,y);
            timeArray1(m) = toc;
        end 
        Metrics = [performance1 trainPerformance1 valPerformance1 testPerformance1];
        sortMetrics = sortrows(Metrics,[1 2],'Ascend');
        performance(i)      = sortMetrics(1,1);
        trainPerformance(i) = sortMetrics(1,2);
        valPerformance(i)   = sortMetrics(1,3);
        testPerformance(i)  = sortMetrics(1,4); 
        timeArray(i) = sum(timeArray1);
        PerformanceTest = [performance trainPerformance valPerformance testPerformance];
        waitbar(i/size(c,1),h,['No of combinations ' num2str(i) ' of ' num2str(size(c,1))],'Name','Calculation process')
    end
        delete(h)
        sc = num2str(c);
        sc = string(sc);
        tiempo = string(datestr(timeArray/86400,'HH:MM:SS.FFF'));
%         results = [results ; PerformanceTest];
        
        if size(c,1) < 2
            Table2 = table(sc,performance,trainPerformance,valPerformance,testPerformance,tiempo,'VariableNames',{'Case','Performance','trainPerformance','valPerformance','testPerformance','Time'});
            Table_Results = Table2;
%             ainets = allnets;
            
        elseif size(c,1) >= 2
            Table2 = table(sc,performance,trainPerformance,valPerformance,testPerformance,tiempo,'VariableNames',{'Case','Performance','trainPerformance','valPerformance','testPerformance','Time'});
            Table_Results = [ Table_Results ; Table2 ];
%             ainets = [ainets ; allnets];
        end  
       
end
nntraintool('close')

WholeTime = sum(minute(Table_Results.Time)+second(Table_Results.Time));
WholeTime = datestr(WholeTime/86400,'HH:MM:SS.FFF');
%
disp('The Whole time is: ')
disp('     ')
disp(WholeTime)

Size = size(Table_Results,1);
idx(1:Size,1) = 1:Size;
Table_Results = addvars(Table_Results,idx,'before','Case');
%% Table of results
% This talbe stores the total tests
% Table_Results

%% Now we show the best ten results
% This table is sort first by performance then by performance test
Best_10_Results = sortrows(Table_Results,{'Performance','testPerformance'},'ascend');
Best_10_Results = head(Best_10_Results,10);
% SizeTableBest = size(Best_10_Results,1);
% 
% for f = 1:length(acvComb)
%     for k = 1:size(acvComb{f})
%         strArray{k} = strjoin(acvComb{k});
%     end
% end

% for l = 1:size(results,1)   
%    TableResults1 = table(results(l,1),results(l,2),results(l,3),results(l,4)); 
%    TablaResults = [TablaResults ; TableResults1]; 
% end