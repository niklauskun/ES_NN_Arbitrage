% input_data=[66.11	68.60	75.37	76.00;
%             65.85	68.21	72.43	72.37;
%             63.53	68.10	75.08	74.87;
%             76.66	75.86	83.88	78.33];
% xvalues = {'DAP0-RTP36','DAP0-RTP288','DAP24-RTP36','DAP24-RTP288'};
% yvalues = {'NYC','LONGIL','NORTH','WEST'};
% h = heatmap(xvalues,yvalues,input_data,'Colormap',winter);
% caxis([60, 90]);
% h.CellLabelFormat = '%.2f';
% h.FontSize = 12;





% lambda = readmatrix('price.csv');
% v = readmatrix('value.csv');
% x=1:288;
% figure
% plot(x,lambda, 'LineWidth', 4);
% hold on
% plot(x,v, 'LineWidth',4);
% hold off
% legend('Real-time Price','Value Function at 50% SoC', 'Location', 'best', 'Orientation','horizontal','FontSize',36)
% set(gca,'FontSize',24)
% xlim([0,288])
% xticks([0 48 96 144 192 240 288])
% xticklabels({'0' '4' '8' '12' '16' '20' '24'})
% xlabel('Hour') 
% ylabel('$/MWh')
% grid on
% box off;




% NYCTR=[78.18
% 79.04
% 79.69
% 80.63
% 77.18
% 74.29
% 79.17
% 80.01
% 79.77
% 75.33];
% NYCTE=[75.57
% 75.74
% 76.53
% 75.37
% 72.82
% 71.67
% 75.08
% 76.73
% 75.01
% 71.68];
% scatter(NYCTR,NYCTE,100,"filled")
% xlabel('Training Profit Ratio (%)') 
% ylabel('Testing Profit Ratio (%)')
% set(gca,'FontSize',24)
xlim([70 86])


% LONGILTR=[78.54
% 81.39
% 80.43
% 80.07
% 75.56
% 72.17
% 79.20
% 79.64
% 79.35
% 77.02];
% LONGILTE=[68.13
% 72.43
% 70.29
% 69.56
% 63.75
% 61.59
% 69.42
% 70.40
% 70.14
% 66.31];
% scatter(LONGILTR,LONGILTE,100,"filled")
% xlabel('Training Profit Ratio (%)') 
% ylabel('Testing Profit Ratio (%)')
% set(gca,'FontSize',24)


% NORTHTR=[78.59
% 80.51
% 79.88
% 80.78
% 80.54
% 78.73
% 77.81
% 80.12
% 79.15
% 80.39];
% NORTHTE=[75.33
% 74.08
% 74.01
% 75.08
% 74.73
% 73.84
% 72.11
% 74.07
% 73.58
% 74.91];
% scatter(NORTHTR,NORTHTE,100,"filled")
% xlabel('Training Profit Ratio (%)') 
% ylabel('Testing Profit Ratio (%)')
% set(gca,'FontSize',24)


% WESTTR=[84.99
% 85.21
% 85.26
% 82.99
% 84.04
% 78.42
% 82.15
% 84.49
% 81.50
% 81.57];
% WESTTE=[83.44
% 83.31
% 83.88
% 81.22
% 81.58
% 75.77
% 80.00
% 82.63
% 79.24
% 79.43];
% scatter(WESTTR,WESTTE,100,"filled")
% xlabel('Training Profit Ratio (%)') 
% ylabel('Testing Profit Ratio (%)')
% set(gca,'FontSize',24)

% y = [66.11 65.85 63.53 76.66; 68.60 68.21 68.10 75.86; 75.37 72.43 75.08 83.88; 76.00 72.37 74.87 78.33; 71.98 65.07 74.60 77.36];
y = [66.11 65.85 63.53 76.66; 68.60 68.21 68.10 75.86; 75.37 72.43 75.08 83.88; 76.00 72.37 74.87 78.33];
bar(y)
ylim([60 90])
ylabel('Profit Ratio (%)')
yticks(60:5:90)
set(gca,'FontSize',24)
legend('NYC','LONGIL','NORTH','WEST', 'Location','best', 'Orientation','horizontal')
% xticks
xticklabels({'Setting 1', 'Setting 2', 'Setting3', 'Setting 4', 'Markov'})
grid on 
box off

% A=[75.37	69.55	76.33	81.36
% 77.23	72.43	77.59	81.90
% 75.41	71.01	75.08	82.64
% 75.89	72.91	77.15	83.88]';
% 
% figure;
% heatmap(A);
% xlabel('Training Locations')
% ylabel('Testing Locations')
% set(gca,'FontSize',24)
% ax = gca;
% ax.XData = [{'NYC','LONGIL','NORTH','WEST'}];
% ax.YData = [{'NYC','LONGIL','NORTH','WEST'}];

% xticklabels({'NYC','LONGIL','NORTH','WEST'})
% yticklabels({'NYC','LONGIL','NORTH','WEST'})