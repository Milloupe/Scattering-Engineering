% bruit.m Genere un bruit gaussien de longueur de correlation L sur 
% un vecteur de taille N.

function z=bruit(L,N)

  M=N;
  d=1;
  nmod=4*M*d/(2*pi*L);
  
  phi=rand(1,nmod)*2*pi;
  A=exp(-2*(pi*L*[1:nmod]/(M*d)).^2);
  z=zeros(1,N);
  for j=1:N
    z(j)=real(sum(exp(i*phi).*A.*exp(2*i*pi*[1:nmod]*j/(M*d))));
  endfor
  z=z*sqrt(max(size(z))/sum(z.^2));
    
%  figure(1)
%  plot(z); 
%  figure(2) 
%  hist(z,30)
% figure(3)
%  auto=zeros(1,N);
%  for j=0:N-1
%    auto(j+1)=sum(z(j+1:N).*z(1:N-j))/(N-j);
%  endfor 
%				plot(auto);  

endfunction