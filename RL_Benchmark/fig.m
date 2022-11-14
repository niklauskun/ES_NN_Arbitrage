M = csvread('NYC_Comp.csv');
MLP = M(:,1)/1000;
RL = M(:,2)/1000;
SDP = M(:,3)/1000;
PF = M(:,4)/1000;

plot(PF,":",'LineWidth',3)
hold on;
plot(MLP,'LineWidth',3)
hold on;
plot(RL,'--','LineWidth',3)
hold on;
plot(SDP,'-.','LineWidth',3)
ylabel('Cumulative Profit (k$)')
xlabel('Time Periods (in 5-min)')
xlim([0 106000])
ylim([0 16])
xticks(0:20000:120000)
yticks(0:2:16)
xticklabels(0:20000:120000)
legend('Perfect Prediction','Our Method', 'RL', 'SDP+MDP', 'location','south')
set(gca,'FontSize', 18)
grid on 
