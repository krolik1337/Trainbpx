clear;  
clc;
data = readmatrix('tae.csv');
p = data(:,1:5);
t = data(:,6);
p=transpose(p);
t=transpose(t);
Pn=mapminmax(p);
Tn=mapminmax(t);
save('tae.mat', 'data', 'p', 't', 'Pn', 'Tn');
% [Ts, ind_Ts] = sort(Tn);                 % posortowanie danych wzorcowych z zapamietaniem ich oryginalnych pozycji
% Pns = zeros(size(Pn));                  % utworzenie nowej macierzy, do ktorej skopiowane zostana posortowane dane uczace
% for i = 1 : length(Ts)              % petla wypelniajaca nowa macierz posortowanymi danymi
%     Pns(: , i) = Pn(: , ind_Ts(i));
% end
% Tn=Ts;
% Pn=Pns;