%% Solve an Autoregression Problem with External Input with NARX Neural
% Input 1: Power, Solar irradiance, Wind Direction, Humidity
% Training Function: trainbr
% initialization: NO
% 26 step ahead
% X9
counter = 0;
rmse_for = 300;
goal_rmse_for = 250;
iterations = 20;
mse_array = zeros(iterations,1);
rmse_array = zeros(iterations,1);
timep = zeros(iterations,1);
h = waitbar(0,'Please wait...');
% set(h,'FaceColor',[1 0 0],'EdgeColor',[0 0 1],'Name','Please wait...');
trainPerformance = zeros(iterations,1);
valPerformance   = zeros(iterations,1);
testPerformance  = zeros(iterations,1);
performance      = zeros(iterations,1);
inPuts = [1 2 3 4];

while ((rmse_for >= goal_rmse_for) & (counter < iterations))
tic  
clc
counter = counter + 1;
Steps = 26;
Delays = 13;

waitbar(counter / iterations,h,['Red neuronal ' num2str(counter) '/' num2str(iterations)]);

input = CleanData(1:3300-Steps,inPuts); %<=========================input
target = CleanData(Steps + 1:3300,1);

X = tonndata(input,false,false);
T = tonndata(target,false,false);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually faster.
% 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';

inputDelays     = 1:Delays;
feedbackDelays  = 1:Delays;
hiddenLayerSize = 10;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

[x,xi,ai,t] = preparets(net,X,{},T);

net.divideFcn = 'divideblock'; % Divide data randomly
net.divideMode = 'time'; % Divide up every sample

if isequal(trainFcn,'trainlm')
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio   = 15/100;
    net.divideParam.testRatio  = 15/100;
else
    net.divideParam.trainRatio = 70/100;    
    net.divideParam.testRatio  = 30/100;
end
% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse'; % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
% net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
%         'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};

% Train the Network
[net,tr] = train(net,x,t,xi,ai);
NET{counter} = net;

% Test the Network
y = net(x,xi,ai);
e = gsubtract(t,y);
performance(counter,1) = perform(net,t,y);

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(t,tr.trainMask);
valTargets   = gmultiply(t,tr.valMask);
testTargets  = gmultiply(t,tr.testMask);
trainPerformance(counter,1) = perform(net,trainTargets,y);
valPerformance(counter,1)   = perform(net,valTargets,y);
testPerformance(counter,1)  = perform(net,testTargets,y);

% clear X;
 l = (3430 - 3300)/Steps + 1;
 Predicted = zeros(Steps,l);
 
for j = 0:l-1   
    End = j*Steps;
    T = CleanData(Steps+1:3300 + End,1);
    T = tonndata(T,false,false);
    wb = waitbar(0,['Iterando...' num2str(j) '/' num2str(l-1)]);
    set(wb,'Name','Progreso del cálculo');
    for i = 1:Steps        
        nets = removedelay(net);
        [xs,xis,ais,ts] = preparets(nets,X,{},T);
        ys = nets(xs,xis,ais);        
        X = CleanData(1:3300 - Steps + End + i,inPuts); %<=========================input
        X = tonndata(X,false,false);
        Predicted(i,j+1) = cell2mat(ys(end));
        T(end+1) = con2seq(Predicted(i,j+1));
        waitbar(i/Steps);
    end  
    delete(wb);
end
   
    Predicted = Predicted(:);
    Real = CleanData(3275:end,1);
    n = size(Real,1);
    
    mse_for = meansqr(Predicted - Real);
    rmse_for = sqrt(mse_for);
    
    mse_array(counter) = mse_for;
    rmse_array(counter) = rmse_for;
    
%     ======================================================================
%     net = init(net); %<=====================================
%     ======================================================================
%     plot(1:n,Predicted,1:n,Real)
%     title(['MSE = ' num2str(mse_for,'%.2e')])
%     legend('Forecasting','Real')
%     nntraintool('close')   

time = toc;

timep(counter) = time; 
end

totaltime = sum(timep);

timep = datestr(timep/86400,'HH:MM:SS.FFF');
totaltime = datestr(totaltime/86400,'HH:MM:SS.FFF');
delete(h);
%% Plot the results

input = CleanData(1:3300-Steps,inPuts); %<=========================input
target = CleanData(Steps + 1:3300,1);

X = tonndata(input,false,false);
T = tonndata(target,false,false);


[~,index] = min(rmse_array);
net = NET{index};

 l = (3430 - 3300)/Steps + 1;
 Predicted = zeros(Steps,l);
 
for j = 0:l-1   
    End = j*Steps;
    T = CleanData(Steps+1:3300 + End,1);
    T = tonndata(T,false,false);
    wb = waitbar(0,['Iterando...' num2str(j) '/' num2str(l-1)]);
    set(wb,'Name','Progreso del cálculo');
    for i = 1:Steps        
        nets = removedelay(net);
        [xs,xis,ais,ts] = preparets(nets,X,{},T);
        ys = nets(xs,xis,ais);        
        X = CleanData(1:3300 - Steps + End + i,inPuts); %<=========================input
        X = tonndata(X,false,false);
        Predicted(i,j+1) = cell2mat(ys(end));
        T(end+1) = con2seq(Predicted(i,j+1));
        waitbar(i/Steps);
    end  
    delete(wb);
end
    delete(h)
    Predicted = Predicted(:);
    Real = CleanData(3275:end,1);
    n = size(Real,1);
    
    mse_for = meansqr(Predicted - Real);
    rmse_for = sqrt(mse_for);
    
    mse_array(counter) = mse_for;
    rmse_array(counter) = rmse_for;
    figure(1)
    plot(1:n,Predicted,1:n,Real)
    title(['MSE = ' num2str(mse_for,'%.2e')])
    legend('Forecasting','Real')
    nntraintool('close')  
    
    % k-fold cross validation performance
%     meanPerformance = mean(performance);
%     stdPerformance = std(performance);
%     TwostdPerformance = 2*stdPerformance;
%     figure(2), hold on
%     plot(performance,'s--b','MarkerSize',10,'LineWidth',2,'MarkerFaceColor','k')
%     xlim([1 20])
%     ylim([min(performance)/1.05 max(performance)*1.05])
%     plot([1 20],[1 1]*meanPerformance,'--r','LineWidth',2)
%     plot([1 20],[1 1]*(meanPerformance + TwostdPerformance),'--k','LineWidth',2)
%     plot([1 20],[1 1]*(meanPerformance - TwostdPerformance),'--k','LineWidth',2)
%     title('20-fold ramdoly cross validation - Performance')
%     xlabel '20-fold '
%     ylabel 'MSE W^2'
%     legend('MSE',['$\mu_{MSE} = $' num2str(meanPerformance,'%.2E')],'$\mu_{MSE} \pm 2 \times \sigma$','Interpreter','Latex')
    
    % k-fold cross validation trainPerformance
%     meantrainPerformance = mean(trainPerformance);
%     stdtrainPerformance = std(trainPerformance);
%     TwostdtrainPerformance = 2*stdtrainPerformance;
%     figure(3), hold on
%     plot(trainPerformance,'s--b','MarkerSize',10,'LineWidth',2,'MarkerFaceColor','k')
%     xlim([1 20])
%     ylim([min(trainPerformance)/1.05 max(trainPerformance)*1.05])
%     plot([1 20],[1 1]*meantrainPerformance,'--r','LineWidth',2)
%     plot([1 20],[1 1]*(meantrainPerformance + TwostdtrainPerformance),'--k','LineWidth',2)
%     plot([1 20],[1 1]*(meantrainPerformance - TwostdtrainPerformance),'--k','LineWidth',2)
%     title('20-fold ramdoly cross validation - Train Performance')
%     xlabel '20-fold '
%     ylabel 'MSE W^2'
%     legend('MSE',['$\mu_{MSE} = $' num2str(meantrainPerformance,'%.2E')],'$\mu_{MSE} \pm 2 \times \sigma$','Interpreter','Latex')
%     
%     % k-fold cross validation valPerformance only with trainlm
%     if isequal(trainFcn,'trainlm')
%         meanValPerformance = mean(valPerformance);
%         stdValPerformance = std(valPerformance);
%         TwostdValPerformance = 2*stdValPerformance;
%         figure(4), hold on
%         plot(valPerformance,'s--b','MarkerSize',10,'LineWidth',2,'MarkerFaceColor','k')
%         xlim([1 20])
%         ylim([min(min(valPerformance)/1.05,(meanValPerformance - TwostdValPerformance)/1.05) max(max(valPerformance)*1.05,(meanValPerformance + TwostdValPerformance)*1.05)])
%         plot([1 20],[1 1]*meanValPerformance,'--r','LineWidth',2)
%         plot([1 20],[1 1]*(meanValPerformance + TwostdValPerformance),'--k','LineWidth',2)
%         plot([1 20],[1 1]*(meanValPerformance - TwostdValPerformance),'--k','LineWidth',2)
%         title('20-fold ramdoly cross validation - Validation Performance')
%         xlabel '20-fold '
%         ylabel 'MSE W^2'
%         legend('MSE',['$\mu_{MSE} = $' num2str(meanValPerformance,'%.2E')],'$\mu_{MSE} \pm 2 \times \sigma$','Interpreter','Latex')
%     end
%      % k-fold cross validation valPerformance only with trainlm
%     meanTestPerformance = mean(testPerformance);
%     stdtestPerformance = std(testPerformance);
%     TwostdtestPerformance = 2*stdtestPerformance;
%     figure(5), hold on
%     plot(testPerformance,'s--b','MarkerSize',10,'LineWidth',2,'MarkerFaceColor','k')
%     xlim([1 20])
%     ylim([min(testPerformance)/1.05 max(testPerformance)*1.05])
%     plot([1 20],[1 1]*meanTestPerformance,'--r','LineWidth',2)
%     plot([1 20],[1 1]*(meanTestPerformance + stdtestPerformance),'--k','LineWidth',2)
%     plot([1 20],[1 1]*(meanTestPerformance - stdtestPerformance),'--k','LineWidth',2)
%     title('20-fold ramdoly cross validation - Test Performance')
%     xlabel '20-fold '
%     ylabel 'MSE W^2'
%     legend('MSE',['$\mu_{MSE} = $' num2str(meanTestPerformance,'%.2E')],'$\mu_{MSE} \pm 2 \times \sigma$','Interpreter','Latex')
    
    