

%% Reading Input
clear all
clc
load('frey_rawface')
d=560;  % Data dimensionality
D=2;  % Latent variable dimensionality
m=200; % Number of nodes in NN
count=500; % Number of images
% convert 560*1 into 20*28 via reshape and then take transpose to get the original image, 
X=zeros(d,count);

%for i=1:count
%   X(1:d,i)=im2double(ff(1:d,i)); 
    
%end

X(1:d,1:100)=im2double(ff(1:d,1:100));
X(1:d,201:300)=im2double(ff(1:d,501:600));
X(1:d,301:400)=im2double(ff(1:d,1001:1100));
X(1:d,401:500)=im2double(ff(1:d,1801:1900));


    
    
%% Backpropagation algorithm
eta=0.001; % Gradient descent
eps=0.01;
Num=(m*(d+1))+(2*D*(m+1))+(m*(D+1))+(2*d*(m+1));  % Number of parameters in NN for frey face dataset
Theta=ones(1,Num)*0.001;
epsilon=normrnd(0,1,D,count);



Theta_prev=0*Theta;
Theta_curr=Theta;
A=[];
i=1;
while abs(Error(Theta_curr,X,epsilon,D,d,m)-Error(Theta_prev,X,epsilon,D,d,m))>eps
    Theta_prev=Theta_curr;
    f=Complete_deriv(Theta_prev,X,epsilon,D,d,m);
    Theta_curr=Theta_prev+eta*f;
    while (Error(Theta_curr,X,epsilon,D,d,m)< Error(Theta_prev,X,epsilon,D,d,m))
       eta=eta/2;
       Theta_curr=Theta_prev+eta*f;
    end
    
    
    A=[A;Error(Theta_curr,X,epsilon,D,d,m)];
    A(i)
    
    eta=eta*2;
    i=i+1;
end

%% Generative Model

Gen_Im=zeros(28*10,20*10);
for i=1:10
    
for j=1:10
epsilon=mvnrnd([0 0],[1 0; 0 1],1)';
Z=epsilon;


L=(m*(d+1))+(2*D*(m+1))+1;                   % Layer 4 to 5
U=(m*(d+1))+(2*D*(m+1))+(m*(D+1));
W5=reshape(Theta_curr(L:U),[m,(D+1)]);
Z5=W5*[1;Z];
a5=tanh(Z5);

L=(m*(d+1))+(2*D*(m+1))+(m*(D+1))+ 1;
U=(m*(d+1))+(2*D*(m+1))+(m*(D+1)) + (2*d*(m+1)) ;   % Layer 5 to Layer 6

W6=reshape(Theta_curr(L:U),[2*d,m+1]);
Z6=W6*[1;a5];
a6=zeros(0,2*d,1);
a6(1:d)=sigmf(Z6(1:d),[1 0]);
a6((d+1):(2*d))=tanh(Z6((d+1):(2*d)));

Mu1=a6(1:d)';                              % Reconstruction loss calculated
Sig1=exp(a6((d+1):(2*d)));
Mu1=a6(1:d)';                              % 6th layer
Sig1=exp(a6((d+1):(2*d)))';
X_hat=Mu1+((Sig1.^0.5).*normrnd(0,1,d,1));
AA=reshape(Mu1,[20,28]);
Gen_Im((1+28*(i-1)):(28*i),(1+20*(j-1)):(20*j))=AA';
end
    
end

imshow(Gen_Im);















