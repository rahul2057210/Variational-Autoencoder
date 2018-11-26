function f=derivat(Theta,x,epsilon,D,d,m)
[W2,W3,W5,W6,a2,a3,a4,a5,a6]=Forw_Prop(Theta,x,epsilon,D,d,m);

% Calculating dErr/da

Mu1=a6(1:d)';                              % 6th layer
Sig1=exp(a6((d+1):(2*d)))';

del_Mu1=-(diag(Sig1)^-1)*Mu1+(diag(Sig1)^-1)*x;
del_Sig1=0.5*(((x-Mu1).^2).*(Sig1.^-1)) -0.5;


del_a6=[del_Mu1;del_Sig1];
C=zeros(2*d,1);
C(1:d)=a6(1:d).*(1-a6(1:d));
C((d+1):(2*d))=(1-a6((d+1):(2*d)).^2);

H1=repmat( (del_a6.*C),1,length(W6(1,:)));
del_a5=sum(H1.*W6)'; % 5th layer
del_a5=del_a5(2:length(del_a5));


H1=repmat((del_a5.*(1-a5.^2)),1,length(W5(1,:))); 
del_a4=sum(H1.*W5)';                           % 4th layer  
del_a4=del_a4(2:length(del_a4));

Mu=a3(1:D);
Sig=exp(a3((D+1):(2*D))); 

del_Mu=del_a4-Mu;
del_Sig=0.5*(del_a4.*(epsilon.*(Sig.^0.5)))+0.5*(1-Sig);
del_a3=[del_Mu;del_Sig];                      % 3rd layer

H1=repmat((del_a3.*(1-a3.^2)),1,length(W3(1,:)));
del_a2=sum(H1.*W3)';                         %  2nd layer
del_a2=del_a2(2:length(del_a2));
% Calculating dErr/dW

H1=repmat([1;a5]',length(W6(:,1)),1);
H2=H1.*repmat(del_a6.*C,1,length(W6(1,:)));
grad_W6=reshape(H2,[1,2*d*(m+1)]);

H1=repmat([1;a4]',length(W5(:,1)),1);
H2=H1.*repmat(del_a5.*(1-a5.^2),1,length(W5(1,:)));
grad_W5=reshape(H2,[1,m*(D+1)]);

H1=repmat([1;a2]',length(W3(:,1)),1);
H2=H1.*repmat(del_a3.*(1-a3.^2),1,length(W3(1,:)));
grad_W3=reshape(H2,[1,2*D*(m+1)]);

H1=repmat([1;x]',length(W2(:,1)),1);
H2=H1.*repmat(del_a2.*(1-a2.^2),1,length(W2(1,:)));
grad_W2=reshape(H2,[1,m*(d+1)]);

f=[grad_W2,grad_W3,grad_W5,grad_W6];



end