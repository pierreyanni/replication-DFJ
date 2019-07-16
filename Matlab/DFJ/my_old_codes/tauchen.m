function [prob,eps,z]=tauchen(N,mu,ro,sig);
% function written down by Adda
% Discretizes an AR(1) process into a Markov chain. Determines the optimal grid 
% and transition matrix. Based on Tauchen (1991).
%
%  y(t)= mu(1-ro) + ro*y(t-1) + u(t)     with   V(u) = sig^2
%
% syntax:
%
% [prob,eps,z]=tauchen(N,mu,ro,sig)
%
% N is the number of points on the grid.
% prob is the transition matrix
% eps contains the cut-off points from - infty to + infty
% z are the grid points, i.e. the conditional mean within [eps(i),eps(i+1)].

global mu_ ro_ sigEps_ sig_ eps_ jindx_

if N==1;prob=1; eps=mu;z=mu;
else;
    if ro==0;
        sigEps=sig;
        eps=repmat(sigEps,[1 N+1]).*repmat(norminv((0:N)/N),size(sigEps))+mu;
        eps(:,1)=-20*sigEps+mu;
        eps(:,N+1)=20*sigEps+mu;
        aux=(eps-mu)./repmat(sigEps,[1 N+1]);
        aux1=aux(:,1:end-1);
        aux2=aux(:,2:end);
        z=N*repmat(sigEps,[1 N]).*(normpdf(aux1)-normpdf(aux2))+mu;
        prob=ones(N,N)/N;
    else;
        sigEps=sig/sqrt(1-ro^2);
        eps=sigEps*norminv((0:N)/N)+mu;
        eps(1)=-20*sigEps+mu;
        eps(N+1)=20*sigEps+mu;
        aux=(eps-mu)/sigEps;
        aux1=aux(1:end-1);
        aux2=aux(2:end);
        z=N*sigEps*(normpdf(aux1)-normpdf(aux2))+mu;
        mu_=mu;ro_=ro;sigEps_=sigEps;eps_=eps;sig_=sig;
        prob=zeros(N,N);

        for i=1:N
            for jindx_=1:N
                prob(i,jindx_)=quadl(@integ3,eps_(i),eps_(i+1),1e-6)*N;
            end
        end
       
    end
 z=z';
 eps=eps';
end


function F=integ3(u);

global mu_ ro_ sigEps_ eps_ jindx_ sig_
aux1=(eps_(jindx_)-mu_*(1-ro_)-ro_*u)/sig_;
aux2=(eps_(jindx_+1)-mu_*(1-ro_)-ro_*u)/sig_;
F=(normcdf(aux2)-normcdf(aux1));
F=F.*exp(-0.5*(u-mu_).*(u-mu_)/sigEps_^2);
pi=4*atan(1);
F=F/sqrt(2*pi*sigEps_^2);
