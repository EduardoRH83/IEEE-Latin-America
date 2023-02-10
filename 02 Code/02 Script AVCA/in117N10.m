%% Solve an Autoregression Problem with External Input with NARX Neural
% Idx 117 number 10: Solar irradiance, wind direction, pressure, dew point, temperature, wind speed

% Training Function: trainlm
% initialization: NO
% 26 step ahead
% X9
%||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
%||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
%||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

input_variable = [2 3 6 7 8 9]; %<---- input vector
TrainFunciton = 'trainbr'; %<-- 'trainlm' ; 'trainbr' 
initialization = 1; % 0 without initialization and 1 with initialization

%||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
%||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
%||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

counter = 0;
rmse_for = 300; %<---300
goal_rmse_for = 250; %<---250
iterations = 20;
mse_array = zeros(iterations,1);
rmse_array = zeros(iterations,1);
timep = zeros(iterations,1);
h = waitbar(0,'Please wait...');
pasos = 1:20;


while ((rmse_for >= goal_rmse_for) & (counter < iterations))
tic  
clc
counter = counter + 1;
Steps = 26;
Delays = 13;

waitbar(pasos(counter) / iterations,h,['Red neuronal ' num2str(counter) '/' num2str(iterations)]);

input = CleanData(1:3300-Steps,input_variable); %<=========================input
target = CleanData(Steps + 1:3300,1);

X = tonndata(input,false,false);
T = tonndata(target,false,false);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually faster.
% 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = TrainFunciton;

inputDelays     = 1:Delays;
feedbackDelays  = 1:Delays;
hiddenLayerSize = 10;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

[x,xi,ai,t] = preparets(net,X,{},T);

net.divideFcn = 'dividerand'; % Divide data randomly
net.divideMode = 'time'; % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;

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
performance = perform(net,t,y);

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(t,tr.trainMask);
valTargets   = gmultiply(t,tr.valMask);
testTargets  = gmultiply(t,tr.testMask);
trainPerformance = perform(net,trainTargets,y);
valPerformance   = perform(net,valTargets,y);
testPerformance  = perform(net,testTargets,y);

% clear X;
 l = (3430 - 3300)/Steps + 1;
 Predicted = zeros(Steps,l);
 
for j = 0:l-1   
    End = j*Steps;
    T = CleanData(Steps+1:3300 + End,1);
    T = tonndata(T,false,false);
%     wb = waitbar(0,['Iterando...' num2str(j) '/' num2str(l-1)]);
%     set(wb,'Name','Progreso del cálculo');
    for i = 1:Steps        
        nets = removedelay(net);
        [xs,xis,ais,ts] = preparets(nets,X,{},T);
        ys = nets(xs,xis,ais);        
        X = CleanData(1:3300 - Steps + End + i,input_variable); %<=========================input
        X = tonndata(X,false,false);
        Predicted(i,j+1) = cell2mat(ys(end));
        T(end+1) = con2seq(Predicted(i,j+1));
%         waitbar(i/Steps);
    end  
%     delete(wb);
end
   
    Predicted = Predicted(:);
    Real = CleanData(3275:end,1);
    n = size(Real,1);
    
    mse_for = meansqr(Predicted - Real);
    rmse_for = sqrt(mse_for);
    
    mse_array(counter) = mse_for;
    rmse_array(counter) = rmse_for;
%==========================================================================
%==========================================================================
%==========================================================================
   if initialization == 1
        net = init(net);
   end 
%==========================================================================
%==========================================================================
%==========================================================================

time = toc;

timep(counter) = time; 
end

totaltime = sum(timep);

timep = datestr(timep/86400,'HH:MM:SS.FFF');
totaltime = datestr(totaltime/86400,'HH:MM:SS.FFF');
delete(h);
%% Plot the results

input = CleanData(1:3300-Steps,input_variable); %<=========================input
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
        X = CleanData(1:3300 - Steps + End + i,input_variable); %<=========================input
        X = tonndata(X,false,false);
        Predicted(i,j+1) = cell2mat(ys(end));
        T(end+1) = con2seq(Predicted(i,j+1));
        waitbar(i/Steps);
    end  
    delete(wb);
end

    Predicted = abs(Predicted(:));
    Real = CleanData(3275:end,1);
    n = size(Real,1);
    
    mse_for = meansqr(Predicted - Real);
    rmse_for = sqrt(mse_for);
    
    mse_array(counter) = mse_for;
    rmse_array(counter) = rmse_for;
    
    plot(1:n,Predicted,1:n,Real)
    title(['MSE = ' num2str(mse_for,'%.2e')])
    legend('Forecasting','Real')
    nntraintool('close')  