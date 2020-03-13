function [W2,W3,W5,W6,a2,a3,a4,a5,a6]=Forw_Prop(Theta,x,epsilon,D,d,m)
global Y;
W2=reshape(Theta(1:(m*(d+1))),[m,d+1]);    % Layer 1 to Layer 2
Z2=W2*[1;x];
a2=tanh(Z2);


L=(m*(d+1))+1;
U=(m*(d+1))+(2*D*(m+1));
W3=reshape(Theta(L:U),[2*D,m+1]);           % Layer 2 to Layer 3
Z3=W3*[1;a2];
a3=tanh(Z3);

Mu=a3(1:D);
Sig=exp(a3((D+1):(2*D)));                    % KL and Latent variable Z calculated
Z=Mu+((Sig.^0.5).*epsilon);
a4=Z;

Z=Y;

L=(m*(d+1))+(2*D*(m+1))+1;                   % Layer 4 to 5
U=(m*(d+1))+(2*D*(m+1))+(m*(D+1));
W5=reshape(Theta(L:U),[m,(D+1)]);
Z5=W5*[1;Z];
a5=tanh(Z5);



L=(m*(d+1))+(2*D*(m+1))+(m*(D+1))+ 1;
U=(m*(d+1))+(2*D*(m+1))+(m*(D+1)) + (2*d*(m+1)) ;   % Layer 5 to Layer 6

W6=reshape(Theta(L:U),[2*d,m+1]);
Z6=W6*[1;a5];
a6=zeros(0,2*d,1);
a6(1:d)=tanh(Z6(1:d));
a6((d+1):(2*d))=tanh(Z6((d+1):(2*d)));

end