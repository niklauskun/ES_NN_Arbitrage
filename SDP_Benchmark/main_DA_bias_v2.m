addpath(genpath('C:\Users\wenmi\Desktop\MarkovESValuation'))

location = 'WEST';
load(strcat('RTP_',location,'_2010_2019.mat'))
load(strcat('DAP_',location,'_2010_2019.mat'))
Ts = 1/12; % time step
Tp = 24/Ts; % number of timepoint
DD = 365; % select days to look back
lastDay = datetime(2019,12,31);
lambda = reshape(RTP(:,(end-DD+1):end),numel(RTP(:,(end-DD+1):end)),1); 
lambda_DA = reshape(DAP(:,(end-DD+1):end),numel(DAP(:,(end-DD+1):end)),1); 
bias = lambda - lambda_DA;
T = numel(lambda); % number of time steps
Nb = 12; % number of bias states (different from pre-processing, use number of states to prevent error)
Gb = 10; % bias state gap

%% load transition matrices
pindep = 0; % price independent, 1 -> True, 0 -> False
pseason = 0; % price seasonal pattern, 1 -> True, 0 -> False
pweek = 0; % price week pattern, 1 -> True, 0 -> False
totalMatrices = 24; %total matrices number in each day
start = 2016;
stop = 2018;

%load expected bias spikes
Ebs = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_expected_bias_spike.csv']));

% load case bias transition matrices
if pindep == 1
    M = zeros(Nb,Nb); % initialize the transition matrices series
    for s = 1:totalMatrices % load matrices
        M(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_null_',sprintf('%d',s-1),'.csv']));
    end
else
    if pseason == 0 && pweek == 0
        M = zeros(Nb,Nb); % initialize the transition matrices series
        for s = 1:totalMatrices % load matrices
            M(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_year_',sprintf('%d',s-1),'.csv']));
        end
    elseif pseason == 1 && pweek == 0
        M1 = zeros(Nb,Nb); % initialize the summer transition matrices series
        M2 = zeros(Nb,Nb); % initialize the non-summer transition matrices series
        for s = 1:totalMatrices % load matrices
            M1(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_summer_',sprintf('%d',s-1),'.csv']));
            M2(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_nonsummer_',sprintf('%d',s-1),'.csv']));
        end
    elseif pseason == 0 && pweek == 1
        M1 = zeros(Nb,Nb); % initialize the weekday transition matrices series
        M2 = zeros(Nb,Nb); % initialize the weekend transition matrices series
        for s = 1:totalMatrices % load matrices
            M1(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_weekday_',sprintf('%d',s-1),'.csv']));
            M2(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_weekend_',sprintf('%d',s-1),'.csv']));
        end
    else
        M1 = zeros(Nb,Nb); % initialize the summer weekday transition matrices series
        M2 = zeros(Nb,Nb); % initialize the summer weekend transition matrices series
        M3 = zeros(Nb,Nb); % initialize the non-summer weekday transition matrices series
        M4 = zeros(Nb,Nb); % initialize the non-summer weekend transition matrices series
        for s = 1:totalMatrices % load matrices
            M1(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_summer_weekday_',sprintf('%d',s-1),'.csv']));
            M2(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_summer_weekend_',sprintf('%d',s-1),'.csv']));
            M3(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_nonsummer_weekday_',sprintf('%d',s-1),'.csv']));
            M4(:,:,s) = readmatrix(join([location,'_',sprintf('%d',start),'_',sprintf('%d',stop),'_',sprintf('%d',Gb),'_','bias_matrix_nonsummer_weekend_',sprintf('%d',s-1),'.csv']));
        end
    end
end

% Define number of SoC samples for testing
% eds = [10 25 50 100 500 1000 2000 5000 10000];
eds = [1000];


% Store profit and computation time for each test
profit_tests = zeros(length(eds),1);
profit_tests_r = zeros(length(eds),1);
profit_tests_d = zeros(length(eds),1);
time_tests = zeros(length(eds),1);

% Start valuation and arbitrage test loop with different SoC granularities
ii = 1;
for ed_test = eds
%%
Pr = .5; % normalized power rating wrt energy rating
P = Pr*Ts; % actual power rating taking time step size into account
eta = 1.0; % efficiency
c = 10; % marginal discharge cost - degradation
ed = 1/ed_test; % SoC sample granularity
ef = 0.0; % final SoC target level, use 0 if none
Ne = floor(1/ed)+1; % number of SOC samples
e0 = 0.0;

% process index
e_index = (0:ed:1)';

% Upsampled process index to use in arbitrage only
e_index_upsampled = (0:1/1000:1)';
% Initialize upsampled value function vector to be used in arbitrage only
vv_upsampled = zeros(length(e_index_upsampled),1);

qEnd = zeros(Ne,Nb,1);  % generate value function samples

qEnd(1:floor(ef*Ne),:) = 1e2; % use 100 as the penalty for final discharge level

%%
tic
q = zeros(Ne,Nb,Tp+1); % initialize the value function series
% v(1,1) is the marginal value of 0% SoC at the beginning of day 1
% V(Ne, T) is the maringal value of 100% SoC at the beginning of the last operating day
q(:,:,end) = qEnd; % update final value function

% value function
v = zeros(Ne,(DD*288+1));

% process index
es = (0:ed:1)';
Ne = numel(es);
% calculate soc after charge vC = (v_t(e+P*eta))
eC = es + P*eta; 
% round to the nearest sample 
iC = ceil(eC/ed)+1;
iC(iC > (Ne+1)) = Ne + 2;
iC(iC < 2) = 1;
% calculate soc after discharge vC = (v_t(e-P/eta))
eD = es - P/eta; 
% round to the nearest sample 
iD = floor(eD/ed)+1;
iD(iD > (Ne+1)) = Ne + 2;
iD(iD < 2) = 1;
% bias index
ba = (-50-Gb/2:Gb:50+Gb/2)';
ba(1) = Ebs(2);
ba(end) = Ebs(1);

%
eS = zeros(T,1); % generate the SoC series
pS = eS; % generate the power series
prS = zeros(T+1,1); % generate the power series
e = e0; % initial SoC

if pindep == 0 && pseason == 1 && pweek == 0
    for d = DD:-1:1
        %% valuation
        q(:,:,end) = q(:,:,1);
        % calculate dates in given year
        date = lastDay - days(DD-d); 
        year_start = datetime(year(date),1,1);
        day = days(date - year_start) + 1;
        if 124 <= day && day <= 284
            tM =M1;
        else
            tM =M2;
        end
        % choose Markov model
        for t = Tp:-1:1
            tp = (d-1)*Tp + t; % current time point
            tH = ceil((t)*Ts); % current hour
            lambdaNode = lambda_DA(tp) + ba;
            for i = 1:Nb
                viE = (tM(i,:,tH) * q(:,:,t+1)')'; % calculate expected value function from next timepoint at price node i
                qo = CalcValueNoUnc(lambdaNode(i), c, P, eta, viE, ed, iC, iD);  % calculate value function at time point t and price node i
%               maximum profit using the current bias discretization
%               ii = int32((Nb-1)/2 + ceil(bias(tp)/Gb)); % get price node i from lambda(t)
%               ii = max(1,min(Nb,ii));
%               qo = CalcValueNoUnc(lambdaNode(ii), c, P, eta, viE, ed, iC, iD);  % calculate value function at time point t and price node i
                q(:,i,t) = qo;
            end
            % value interpolation if have 0 probability node
            if nnz(sum(tM(:,:,tH),2)) ~= Nb
                for i = 1:Nb
                    if sum(tM(i,:,tH))==0
                        q(49:end,i,t) =NaN;
                    end
                end
                q(:,:,t) = fillmissing(q(:,:,t),'nearest',2);
            end
        end
        %% abitrage
        for t = 1:Tp % start from the first day and move forwards
            tp = (d-1)*Tp + t; % current time point
            tH = ceil((t)*Ts); % current hour
            i = int32((Nb-1)/2 + ceil(bias(tp)/Gb)); % get price node i from lambda(t)
            i = max(1,min(Nb,i));
            v(:,tp) = (tM(i,:,tH) * q(:,:,t+1)')';
        end
        fprintf('Day = %d\n', d)
    end
elseif pindep == 0 && pseason == 0 && pweek == 1
    for d = DD:-1:1
        %% valuation
        q(:,:,end) = q(:,:,1);
        % calculate dates in given year
        date = lastDay - days(DD-d); 
        if isweekend(date) == 0
            tM =M1;
        else
            tM =M2;
        end
        % choose Markov model
        for t = Tp:-1:1
            tp = (d-1)*Tp + t; % current time point
            tH = ceil((t)*Ts); % current hour
            lambdaNode = lambda_DA(tp) + ba;
            for i = 1:Nb
                viE = (tM(i,:,tH) * q(:,:,t+1)')'; % calculate expected value function from next timepoint at price node i
                qo = CalcValueNoUnc(lambdaNode(i), c, P, eta, viE, ed, iC, iD);  % calculate value function at time point t and price node i
%               maximum profit using the current bias discretization
%               ii = int32((Nb-1)/2 + ceil(bias(tp)/Gb)); % get price node i from lambda(t)
%               ii = max(1,min(Nb,ii));
%               qo = CalcValueNoUnc(lambdaNode(ii), c, P, eta, viE, ed, iC, iD);  % calculate value function at time point t and price node i
                q(:,i,t) = qo;
            end
            % value interpolation if have 0 probability node
            if nnz(sum(tM(:,:,tH),2)) ~= Nb
                for i = 1:Nb
                    if sum(tM(i,:,tH))==0
                        q(:,i,t) =NaN;
                    end
                end
                q(:,:,t) = fillmissing(q(:,:,t),'nearest',2);
            end
        end
        %% abitrage
        for t = 1:Tp % start from the first day and move forwards
            tp = (d-1)*Tp + t; % current time point
            tH = ceil((t)*Ts); % current hour
            i = int32((Nb-1)/2 + ceil(bias(tp)/Gb)); % get price node i from lambda(t)
            i = max(1,min(Nb,i));
            v(:,tp) = (tM(i,:,tH) * q(:,:,t+1)')';
        end
        fprintf('Day = %d\n', d)
    end
elseif pindep == 0 && pseason == 1 && pweek == 1
    for d = DD:-1:1
        %% valuation
        q(:,:,end) = q(:,:,1);
        for t = Tp:-1:1
            tp = (d-1)*Tp + t; % current time point
            tH = ceil((t)*Ts); % current hour
            lambdaNode = lambda_DA(tp) + ba;
            for i = 1:Nb
                viE = (tM(i,:,tH) * q(:,:,t+1)')'; % calculate expected value function from next timepoint at price node i
                qo = CalcValueNoUnc(lambdaNode(i), c, P, eta, viE, ed, iC, iD);  % calculate value function at time point t and price node i
%               maximum profit using the current bias discretization
%               ii = int32((Nb-1)/2 + ceil(bias(tp)/Gb)); % get price node i from lambda(t)
%               ii = max(1,min(Nb,ii));
%               qo = CalcValueNoUnc(lambdaNode(ii), c, P, eta, viE, ed, iC, iD);  % calculate value function at time point t and price node i
                q(:,i,t) = qo;
            end
        end  
        %% abitrage
        for t = 1:Tp % start from the first day and move forwards
            tp = (d-1)*Tp + t; % current time point
            tH = ceil((t)*Ts); % current hour
            i = int32((Nb-1)/2 + ceil(bias(tp)/Gb)); % get price node i from lambda(t)
            i = max(1,min(Nb,i));
            v(:,tp) = (tM(i,:,tH) * q(:,:,t+1)')';
        end
        fprintf('Day = %d\n', d)
    end
else
    for d = DD:-1:1
        %% valuation
        q(:,:,end) = q(:,:,1);
%       q(:,end) = qEnd; 
        for t = Tp:-1:1
            tp = (d-1)*Tp + t; % current time point
            tH = ceil((t)*Ts); % current hour
            lambdaNode = lambda_DA(tp) + ba;
            for i = 1:Nb
                viE = (M(i,:,tH) * q(:,:,t+1)')'; % calculate expected value function from next timepoint at price node i
                qo = CalcValueNoUnc(lambdaNode(i), c, P, eta, viE, ed, iC, iD);  % calculate value function at time point t and price node i
%                 qo = CalcValueNoUnc_Charge(lambdaNode(i), c, P, eta, viE, ed, iC, iD);  % calculate value function at time point t and price node i
%               maximum profit using the current bias discretization
%               ii = int32((Nb-1)/2 + ceil(bias(tp)/Gb)); % get price node i from lambda(t)
%               ii = max(1,min(Nb,ii));
%               qo = CalcValueNoUnc(lambdaNode(ii), c, P, eta, viE, ed, iC, iD);  % calculate value function at time point t and price node i
                q(:,i,t) = qo;
            end
        end
        %% abitrage
        for t = 1:Tp % start from the first time period to last time period
            tp = (d-1)*Tp + t; % current time point
            tH = ceil((t)*Ts); % current hour
            i = int32((Nb-1)/2 + ceil(bias(tp)/Gb)); % get price node i from lambda(t)
            i = max(1,min(Nb,i));
            v(:,tp) = (M(i,:,tH) * q(:,:,t+1)')';
        end
        %fprintf('Day = %d\n', d)
    end   
end


%% abitrage
for d = 1:DD
    for t = 1:Tp % start from the first day and move forwards
        tp = (d-1)*Tp + t; % current time point
        
        vv = v(:,tp); % read the SoC value for this day
        vv_upsampled = interp1(e_index, vv, e_index_upsampled);
        vv = vv_upsampled;

        [e, p] =  Arb_Value(lambda(tp), vv, e, P, 1, eta, c, size(vv,1));
%       [e, p] =  Arb_Value_Charge(lambda(tp), v(:,tp+1), e, P, 1, eta, c, size(v,1));
        eS(tp) = e; % record SoC
        pS(tp) = p; % record Power
        prS(tp+1) = prS(tp) + p*lambda(tp) - c*max(0,p);
    end
end

ProfitOut = sum(pS.*lambda) - sum(c*pS(pS>0));
Revenue = sum(pS.*lambda);
Discharge = sum(pS(pS>0));


fprintf('Profit = %e, Revenue = %e, Discharged = %e\n', ProfitOut, Revenue, Discharge);

solTimeOut = toc;

profit_tests(ii) = ProfitOut;
profit_tests_r(ii) = Revenue;
profit_tests_d(ii) = Discharge;

time_tests(ii) = solTimeOut;
ii = ii + 1;


end

%% Plot
% close all
% figure
% semilogx(eds, profit_tests, LineWidth=2)
% hold on
% semilogx(eds, 12593*ones(length(eds),1), LineWidth=2,LineStyle="--")
% ylim([0 15000])
% figure
% loglog(eds, time_tests, LineWidth=2)



%%

% % save price and dispatch as csv
% Table1 = array2table([lambda_DA lambda]);
% Table1.Properties.VariableNames(1:2) = {'DAP','RTP'};
% writetable(Table1, 'price.csv')
% 
% Charge = max(zeros(T,1),pS);
% Discharge = max(zeros(T,1),-pS);
% Table2 = array2table([Charge Discharge]);
% Table2.Properties.VariableNames(1:2) = {'Charge','Discharge'};
% writetable(Table2, 'dispatch_MarkovDA1618.csv')

% for t = T:-1:1 % start from the last timepoint and move backwards
%     vi = v(:,t+1); % input value function from next time point
%     
%     i = floor(bias(t)/Gb)+1;
%     if i < -(Nb-1)/2
%         i = -(Nb-1)/2;
%     elseif i > (Nb-1)/2
%         i = (Nb-1)/2;
%     end
%     i = i + (Nb - 1)/2 + 1;
%     
%     if pindep == 0 && ((pseason == 1 && pweek == 0) || (pseason == 0 && pweek == 1))
%         date = lastDay - days(ceil((T-t+1)/(T/(DD+1))) - 1);
%         year_start = datetime(year(date),1,1);
%         day = days(date - year_start) + 1;
%         %choose transition matrix for timepoint t
%         if mod(ceil(t*Ts), totalMatrices) == 0
%             tM1 = M1(:,:,totalMatrices); %when mod=0, use last transition matrix
%             tM2 = M2(:,:,totalMatrices); %when mod=0, use last transition matrix
%         else
%             tM1 = M1(:,:,mod(ceil(t*Ts),totalMatrices)); %when mode!=0
%             tM2 = M2(:,:,mod(ceil(t*Ts),totalMatrices)); %when mode!=0
%         end
%         %calculate expected value function at timepoint t, price node i
%         biasE = 0;
%         if pseason == 1 && pweek == 0
%             for j = 1:Nb
%                 biasE = biasE + ((124 <= day && day <= 284) * tM1(i,j) + (124 > day | day > 284) * tM2(i,j)) * (j - (Nb - 1)/2 - 1) * Gb; % calculate expected value function from next timepoint at price node i
%             end
%         else
%             for j = 1:Nb
%                 biasE = biasE + ((isweekend(date) == 0) * tM1(i,j) + (isweekend(date) == 1) * tM2(i,j)) * (j - (Nb - 1)/2 - 1) * Gb; % calculate expected value function from next timepoint at price node i
%             end
%         end
%         lambdaNode = lambda_DA(t) + biasE; % calculate expected price at price node i
%         vo = CalcValueNoUnc(lambdaNode, c, P, eta, vi, ed, iC, iD);  % calculate value function at time point t and price node i
%         v(:,t) = vo; % record the result  
%     elseif pindep == 0 && pseason == 1 && pweek == 1
%         %choose transition matrix for timepoint t
%         if mod(ceil(t*Ts), totalMatrices) == 0
%             tM1 = M1(:,:,totalMatrices); %when mod=0, use last transition matrix
%             tM2 = M2(:,:,totalMatrices); %when mod=0, use last transition matrix
%             tM3 = M3(:,:,totalMatrices); %when mod=0, use last transition matrix
%             tM4 = M4(:,:,totalMatrices); %when mod=0, use last transition matrix
%         else
%             tM1 = M1(:,:,mod(ceil(t*Ts),totalMatrices)); %when mode!=0
%             tM2 = M2(:,:,mod(ceil(t*Ts),totalMatrices)); %when mode!=0
%             tM3 = M3(:,:,mod(ceil(t*Ts),totalMatrices)); %when mode!=0
%             tM4 = M4(:,:,mod(ceil(t*Ts),totalMatrices)); %when mode!=0
%         end
%         %calculate expected bias
%         biasE = 0;
%         for j = 1:Nb
%             biasE = biasE + tM(i,j) * (j - (Nb - 1)/2 - 1) * Gb; % calculate expected value function from next timepoint at price node i
%         end
%         lambdaNode = lambda_DA(t) + biasE; % calculate expected price at price node i
%         vo = CalcValueNoUnc(lambdaNode, c, P, eta, vi, ed, iC, iD);  % calculate value function at time point t and price node i
%         v(:,t) = vo; % record the result  
%     else
%         %choose transition matrix for timepoint t
%         if mod(ceil(t*Ts), totalMatrices) == 0
%             tM = M(:,:,totalMatrices); %when mod=0, use last transition matrix
%         else
%             tM = M(:,:,mod(ceil(t*Ts),totalMatrices)); %when mode!=0
%         end
%         %calculate expected value function
%         for j = 1:Nb
%             lambdaNode = lambda_DA(t) + (j - (Nb - 1)/2 - 1) * Gb; % calculate expected price at price node i
%             vo = CalcValueNoUnc(lambdaNode, c, P, eta, vi, ed, iC, iD);  % calculate value function at time point t and price node i
%             v(:,t) = v(:,t) + tM(i,j) * vo; % record the result     
%         end
%     end
% end

% tElasped = toc;

% %% perform the actual arbitrage
% eS = zeros(T,1); % generate the SoC series
% pS = eS; % generate the power series
% e = e0; % initial SoC
% 
% for d = 1:DD
%     for t = 1:T % start from the first day and move forwards
%         i = int32(1) * int32(bias(t) < -50) + ...
%             int32(Nb) * int32(bias(t) >= 50) + ...        
%             (idivide(bias(t),int32(Gb)) + 2) * int32(bias(t) >= -50 && lambda(t) < 50); %get price node i from lambda(t)
%         tp = mod(t,int32(Tp));    
%         vv = C{d,i}(:,Tp+1) * (tp == 0) + C{d,i}(:,tp+1)* (tp ~= 0);
%         [e, p] =  Arb_Value(lambda(t), vv, e, P, 1, eta, c, size(v,1));
%         eS(t) = e; % record SoC
%         pS(t) = p; % record Power
%     end
% end
% 
% ProfitOut = sum(pS.*lambda) - sum(c*pS(pS>0));
% Revenue = sum(pS.*lambda);
% Discharge = sum(pS(pS>0));
% 
% fprintf('Profit=%e, revenue=%e, total discharge=%e',ProfitOut, Revenue, Discharge)
% 
% solTimeOut = toc;
