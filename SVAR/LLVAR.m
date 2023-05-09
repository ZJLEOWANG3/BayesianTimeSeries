function [LL]=LLVAR(theta,Y,Z,n,p)
T=length(Z);
C=reshape(theta(1:n^2),n,n);

A=reshape(theta(n^2+1:end),n,n*p);
Omega=C*C';
Omegainv=eye(n)/Omega;
res= Y-A*Z;
LL=0; % init log-likelihood
ldO = log(det(Omega));
for t=1:T-1 % compute log-likelihood for each time period in the data 
LL=LL-0.5*(ldO + res(:,t)'*Omegainv*res(:,t)); % the likelihood function of a state space model in Slides 10 P 9
end
