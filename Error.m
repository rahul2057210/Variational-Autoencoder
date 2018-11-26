function f=Error(Theta,X,epsilon,D,d,m)
Err=0;
for i=1:length(X(1,:))
   Err=Err+NN(Theta,X(:,i),epsilon(:,i),D,d,m); 
    
end
f=Err;

end