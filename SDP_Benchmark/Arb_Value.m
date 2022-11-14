function [eF, pF] = Arb_Value(lambda, v,e,P,E,eta,c,N)
%%
% Title: Arbitrage test using value function





%%
iE = ceil((N-1)*e/E)+1; % find the nearest SoC index


vF = v; % read the value function

vF((iE+1):end) = vF((iE+1):end)*eta; % charge efficiency
vF(1:(iE-1)) = (vF(1:(iE-1)))/eta + c ; % discharge efficiency

% charge index
iC = find(vF >= lambda, 1, 'last');

% discharge index
iD = find(vF <= lambda, 1, 'first');

% iF = iC*(iC > iE) + iD*(iD < iE) + iE*(iC <= iE)*(iD >= iE);
if iC > iE
    iF = iC;
elseif iD < iE
    iF = iD;
else
    iF = iE;
end

eF = (iF-1)/(N-1)*E;

eF = max(min(eF, e + P*eta), e-P/eta);

pF = (e-eF)/eta*((e-eF) < 0) + (e-eF)*eta*((e-eF) > 0);