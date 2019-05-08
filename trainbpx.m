clear; 
close all; 
clc;
format compact;
% load parkinson
data = readmatrix('tae.csv');
p = data(:,1:5);
t = data(:,6);
p=transpose(p);
t=transpose(t);
Pn=mapminmax(p);
Tn=mapminmax(t);
Pnt=Pn(:, 136:151);%dane testowe
Tnt=Tn(:, 136:151);
Pn=Pn(:, 1:135);%dane ucz¹ce
Tn=Tn(:, 1:135);

[Ts, ind_Ts] = sort(Tn);                 % posortowanie danych wzorcowych z zapamietaniem ich oryginalnych pozycji
Pns = zeros(size(Pn));                  % utworzenie nowej macierzy, do ktorej skopiowane zostana posortowane dane uczace
for i = 1 : length(Ts)              % petla wypelniajaca nowa macierz posortowanymi danymi
    Pns(: , i) = Pn(: , ind_Ts(i));
end

% Tn=Ts;
% Pn=Pns;
%Pn
%T
% sprawdziæ jaka jest u¿ywana miara b³êdu, tzn. sse (0.25) czy mse(0.25/150)

S1_vec=5:5:30; %wynik badania przesiewowego
S2_vec=S1_vec; 
lr_inc_vec = 1.01:0.02:1.09;
lr_dec_vec = 0.5:0.1:0.9;
er_vec = 1.01:0.015:1.07;
mc_vec = [0.1:0.2:0.9 0.95 0.99];
PK_v6=zeros(length(S1_vec),length(S2_vec),length(lr_inc_vec),length(lr_dec_vec),length(er_vec),length(mc_vec));
SSE_v6=PK_v6;

for ind_S1=1:length(S1_vec)
for ind_S2=1:ind_S1 
for ind_lr_inc=1:length(lr_inc_vec)
for ind_lr_dec=1:length(lr_dec_vec)
for ind_er=1:length(er_vec)
for ind_mc=1:length(mc_vec)
                        
net = feedforwardnet([S1_vec(ind_S1) S2_vec(ind_S2)],'traingdx');
net.trainParam.lr = 0.05; % wpisaæ warto?æ koñcowš po 20000 iteracji
net.trainParam.epochs = 100; %
net.trainParam.goal = 0.25/135; %liczba rekordow 
net.trainParam.lr_inc = lr_inc_vec(ind_lr_inc); %1.05;
net.trainParam.lr_dec = lr_dec_vec(ind_lr_dec); % 0.7
net.trainParam.max_perf_inc = er_vec(ind_er); % 1.04
net.trainParam.mc = mc_vec(ind_mc); % 0.9
net.trainParam.max_fail =50;
[net,tr] = train(net,Pn,Ts);
y = net(Pn); %(??) 

%PK = (1-sum((abs(T-y )>=.5)')/length(T))*100; %
PK_v6(ind_S1, ind_S2, ind_lr_inc, ind_lr_dec, ind_er,ind_mc) = (1-sum((abs(Ts-y )>=.5))/length(Ts))*100;
%SSE_v5=PK_v5;
SSE_v6(ind_S1, ind_S2, ind_lr_inc, ind_lr_dec, ind_er, ind_mc) = tr.perf(end); 
%SSE_v5 = tr(2,end); %???

                    
end
end
end
end
end
end