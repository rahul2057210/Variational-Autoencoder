function f=Complete_deriv(Theta,X,epsilon,D,d,m)

Deriv=zeros(1,length(Theta));
for i=1:length(X(1,:))
   Deriv=Deriv+derivat(Theta,X(:,i),epsilon(:,i),D,d,m); 
    
end
f=Deriv;

end