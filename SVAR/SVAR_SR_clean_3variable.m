% Question: (1) does it mean I need to draw initial thetaA from improper
% uniform? If I use unifrnd(-inf,inf,[n,n*p]) is returning all NAN. I do
% not need to impose it in thetaB and MCMC right? 
%%% So does it mean Q1 we did that for three conditions?
% not sure about the order; the row should be variables; columns are the
% shock
% how to set non-negative compared to positive 0, inf

% Estimate SVAR(4) using Metropolis-Hastings
clc
clear all
close all

%set control parameters of code 
n=3;%number of variales in VAR
p=1;%nuber of lags in VAR
J=1e5;%Number of draws/size of MCMC
S=1e4;%Number of independent draws from MCMC for figures
Q4=0;%whether to perform Q4, if 1, then no restrictions on FFR

%load and construct data
load macro_data_r_pi_y
load template
% template.FigSize = [10,4.3750];
% template.PaperSize = [15,6];

% Y = A*Z + B*u
% shape-wise: n x T = n x (n x lag) x (n x lag) x T + (n x n) x (n x T)
% n = variable, T = time series - lag
Z=macro_data_r_pi_y; % size of 3 x 160

Z_raw=Z;

% lag of 4, so the end-1 -4 +1 should be end-4
Znew = [];
for tempi=1:p
Zi=Z(:,-tempi+p+1:end-tempi);
Znew=[Znew;Zi];
end
Z = Znew;
Y=Z_raw(:,(p+1):end);

%impose sign restrictions boundaries of as uniform prior densities; this is
%from order for each folumn in B
if Q4==1 % no sign restriction on FFR to demand and supply shocks
LB_B=[0,-inf,-inf, -inf,0,-inf, -inf,0,0;];
UB_B=[inf,0,0, inf,inf,0, inf,inf,inf;];
else
LB_B=[0,-inf,-inf, 0,0,-inf, 0,0,0;];
UB_B=[inf,0,0, inf,inf,0, inf,inf,inf;];
end

% if p==4
% Z1=Z(:,4:end-1);
% Z2=Z(:,3:end-2);
% Z3=Z(:,2:end-3);
% Z4=Z(:,1:end-4);
% Z=[Z1;Z2;Z3;Z4;]; % shape of (lag 4 * 3 variable) x (160 - lag 4); 12 x 156
% Y=Z_raw(:,5:end); % 3 x 156; variable x (time series - lag)
% elseif p==1
% Z=Z(:,1:end-1);
% Y=Z_raw(:,2:end);
% end

Z=Z-mean(Z,2); % center at each column
Y=Y-mean(Y,2);
T=max(size(Z)); % time series of 160 - lag of 4
%%
%Use OLS starting values for MCMC
A_ols= Y*Z'/(Z*Z'); % closed form ' denotes transpose, /X denotes inverse of X; size of 3 x 12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %A_ols=unifrnd(-inf,inf,[n,n*p]);
theta_A=A_ols(:); % to 1d
resid=Y-A_ols*Z; % get residual 
diag_omega=diag(cov(resid')); % get the diagonal of covariance matrix of residual (n x n), then 1 x n; which is variance
B=diag(diag_omega.^.5); % get the std and then make it a diagonal matrix; n x n
theta_B=B(:); % n^2

% Since theta_B is derived from covariance matrix, so all dig elements >=0
% and other elements are 0
% By imposing required sign restriction for Q1 and Q4, we only need to impose first element >0
if theta_B(1,1)<=0
   theta_B(1,1)=0.3895;
end

theta=[theta_B;theta_A]; %Starting value for parameter vector for both B and A; size of theta: (n x n x lag + n x n)

% Set control parameters of MH define needed objects 
epseye=1e-5;%initial scaling of proposal distribution covariance matrix
lpostdraw = -9e+200;%Set up so that the first candidate draw is always accepted
bdraw=theta;%mean of first draw
vscale=diag(abs(theta))*epseye+1e-6*eye(length(theta)); %variance of first draw; size of length(theta)^2

%Store all draws in MCMC
MCMC=zeros(length(theta),J);
MCMC(:,1)=theta;
%Keep track of switches and drwas outside LB and UB
OutsideProp=zeros(J,1);
SwitchesProp=zeros(J,1);
%Number of draws outside parameter boundaries
q=0;
%Number of switches (acceptances)
pswitch=0;
%Iteration counter
iter=0;

%--------------------------------------------------------------------------
% MH algorithm starts here
%--------------------------------------------------------------------------
tic
for iter=1:J
    
    % Draw from proposal density Theta*_{t+1} ~ N(Theta_{t},vscale)
    bcan = bdraw + norm_rnd(vscale);
    %bcan = bdraw + mvnrnd(zeros(size(vscale)),vscale);
    if min(bcan(1:n^2) >= LB_B')==1 %impose sign restrictions
        if min(bcan(1:n^2) <= UB_B')==1 %impose sign restrictions
           % it is not allowed to separate parameteres A and C; that is, we
           % accept all or reject all
           lpostcan = LLVAR(bcan,Y,Z,n,p);%returns log-likelihood of Y=BZ+e where Y is nxT and Z is (nxp)xT
           laccprob = lpostcan-lpostdraw; %already imposed improper uniform distribution on prior, which is set as zero
           
        else
            laccprob=-9e+200;
            q=q+1;
        end
    else
        laccprob=-9e+200;
        q=q+1;
    end
    
    %Accept candidate draw with log prob = laccprob, else keep old draw
    if log(rand)<laccprob
        lpostdraw=lpostcan;
        bdraw=bcan;
        pswitch=pswitch+1;
    end
    
    MCMC(:,iter)=bdraw;%save current draw
    
    OutsideProp(iter)=q/iter;
    SwitchesProp(iter)=pswitch/iter;
    
    %use adaptive propsoal density
    if iter >= 1000 && mod(iter,20000)==0
    vscale=2e-1*cov(MCMC(:,1:100:iter)');
    iter
    
    end
    
end
toc
MCMC=MCMC(:,J*.5:end); % remove first half of the MCMC draws
disp(['iter: ',num2str(iter)]);
disp(['acceptance rate: ',num2str(SwitchesProp(iter))]);


%%

%Define objects needed for figures

periods=20;%periods for IRFs
ra = J/2;
ff = ceil(ra.*rand(S,1)); %contains indices of S independent draws from latter MCMC; 
% rand is standard uniform distribution on the open interval(0,1)

%construct matrices to store output in
RR=zeros(n,T,S);
SS=zeros(n,T,S);
corr_emp_news_b=zeros(1,S);
corr_emp_news_n=zeros(1,S);

corr_stk_news_b=zeros(1,S);
corr_stk_news_n=zeros(1,S);
IMPR=zeros(n,n,periods,S);

BB=zeros(n*p,n);
for s=1:S
    %objects needed for IRFs
B_draw=reshape(MCMC(1:length(theta_B),ff(s)), n,n); % randomly picked S draws from latter half MCMC draws
A_draw=reshape(MCMC((length(theta_B)+1):end,ff(s)), n,n*p);
    
AA = A_draw;
for addAA=0:p-2
    AA = [AA;
        zeros(n,n*addAA),eye(n),zeros(n,n*p-n*(addAA+1))];
end

BB(1:n,:)=B_draw;


%objects needed for correlation histograms
RR(:,:,s) = Y - A_draw*Z;
SS(:,:,s) = B_draw\RR(:,:,s);


  %compute IRFs 
  AAss = AA^0; % make the matrix equals to I
for ss=0:periods-1     
    for ni=1:n
        IMPR(:,ni,ss+1,s)= [eye(n),zeros(n,n*p-n)]*(AAss)*BB(:,ni); % variable ni
    end
    AAss = AAss*AA;
end


end

% save impsort data
Impsort = containers.Map();
for i=1:n
    for j=1:n
    mapid = sprintf("%i,%i",i,j);
    Impsort(mapid) = reshape(sort(IMPR(i,j,:,:),4),periods,S);
    end
end

% density and histogram for all parameters
% plot Q1-C
figure
count=1;
for i=1:n^2
subplot(n,n,count);
ksdensity(MCMC(i,:));
xlabelstr=sprintf("C %i",count);
xlabel(xlabelstr,'fontsize',16)
count=count+1;
end
setprinttemplate(gcf,template)
saveas(gcf,sprintf('SR_PS2_Q1_C_lag%i.pdf',p))

% plot Q1-A
for j=1:p
figure
count=1;
for i=j*n^2+1 : n^2 * (j+1)
subplot(n,n,count);
ksdensity(MCMC(i,:));
xlabelstr=sprintf("A%i %i",j,count);
xlabel(xlabelstr,'fontsize',16)
count=count+1;
end
setprinttemplate(gcf,template)
saveas(gcf,sprintf('SR_PS2_Q1_A%i_lag%i.pdf',j,p))
end


% IRFs
%define plotted percentiles of posterior
med=0.5;
top=0.95;
top16=0.84;
bot16=0.16;
bot=.05;
%%
variable=["FFR","Inflation","GDP Growth"];
shock=["monetary policy shock","supply shock","demand shock"];

%plot IRFs
x=[1:1:periods];
figure
count=1;
for i=1:n
    for j=1:n
    subplot(n,n,count);
    hold on;
    mapid = sprintf("%i,%i",i,j);
    Impsortij = Impsort(mapid);
    plot(Impsortij(:,S*med),'color','black','LineWidth',2);
    hold on;
    plot(Impsortij(:,S*top),'color','black','LineWidth',1,'linestyle',':');
    hold on;
    plot(Impsortij(:,S*bot),'color','black','LineWidth',1,'linestyle',':');
    xlabelstr = sprintf('Response of \n %s to %s',variable(i),shock(j));
    xlabel(xlabelstr,'fontsize',6)

    count=count+1;
    end
end

setprinttemplate(gcf,template)
saveas(gcf,sprintf('SR_PS2_Q3_lag%i.pdf',p))

% save PS2-Q1


% plot A1 and C
x = 1:(J/2+1);
% Plot C
C = MCMC(1:n^2,:);
count=1;
figure
for i=1:n^2
subplot(n,n,count)
plot(x,C(i,:),"color","black","LineWidth",2);
title(sprintf("C %i",i))
count=count+1;
end
setprinttemplate(gcf,template)
saveas(gcf,sprintf('SR_PS2_Q2_C_lag%i.pdf',p))

% Plot A1
A1 = MCMC((n^2+1):(2*n^2),:);
count=1;
figure
for i=1:n^2
subplot(n,n,count)
plot(x,A1(i,:),"color","blue","LineWidth",2);
title(sprintf("A1 %i",i))
count=count+1;
end
setprinttemplate(gcf,template)
saveas(gcf,sprintf('SR_PS2_Q2_A1_lag%i.pdf',p))


