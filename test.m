clear; 
close all; 
clc;
format compact;
data = readmatrix('tae.csv');
p = data(:,1:5);
t = data(:,6);
p=transpose(p);
Tn=transpose(t);
Pn=mapminmax(p);
%Tn=mapminmax(t);
Pnt=Pn(:, 136:151);%dane testowe
Tnt=Tn(:, 136:151);
Pn=Pn(:, 1:135);%dane ucz¹ce
Tn=Tn(:, 1:135);

[Ts, ind_Ts] = sort(Tn);                 % posortowanie danych wzorcowych z zapamietaniem ich oryginalnych pozycji
Pns = zeros(size(Pn));                  % utworzenie nowej macierzy, do ktorej skopiowane zostana posortowane dane uczace
for i = 1 : length(Ts)              % petla wypelniajaca nowa macierz posortowanymi danymi
    Pns(: , i) = Pn(: , ind_Ts(i));
end

Tn=Ts;
Pn=Pns;
%Pn
%T
% sprawdziæ jaka jest u¿ywana miara b³êdu, tzn. sse (0.25) czy mse(0.25/150)

S1_vec=20; %wynik badania przesiewowego
S2_vec=10; 
lr_inc_vec = 1.01;
lr_dec_vec = 0.9;
er_vec = 1.01;
mc_vec = 0.99;

                        
net = feedforwardnet([60 40],'traingdx');
net.trainParam.lr = 0.01; % wpisaæ warto?æ koñcowš po 20000 iteracji
net.trainParam.epochs = 50000; %
net.trainParam.goal = 0.25/135; %liczba rekordow 
net.trainParam.lr_inc = lr_inc_vec; %1.05;
net.trainParam.lr_dec = lr_dec_vec; % 0.7
net.trainParam.max_perf_inc = er_vec; % 1.04
net.trainParam.mc = mc_vec; % 0.9
net.trainParam.max_fail =5000;
[net,tr] = train(net,Pn,Tn);
y = net(Pn); %(??) 

%PK = (1-sum((abs(T-y )>=.5)')/length(T))*100; %
PK_v6 = (1-sum((abs(Tn-y )>=.5))/length(Tn))*100;
%SSE_v5=PK_v5;
SSE_v6 = tr.perf(end); 
%SSE_v5 = tr(2,end); %???

                    