clear; 
close all; 
clc;
format compact;
load tae;
% sprawdziæ jaka jest u¿ywana miara b³êdu, tzn. sse (0.25) czy mse(0.25/150)

S1_vec=3:3:30;
S2_vec=S1_vec; 
lr_inc_vec = 1.01:0.02:1.09;
lr_dec_vec = 0.5:0.1:0.9;
er_vec = 1.01:0.015:1.07;
mc_vec = [0.1:0.2:0.9 0.95 0.99];%, length(lr_inc_vec),length(lr_dec_vec),length(er_vec),length(mc_vec)
PK_v6 = zeros(length(S1_vec),length(S2_vec));
SSE_v6 = PK_v6;
epochs_v6 = SSE_v6;

for ind_S1=1:length(S1_vec)
    for ind_S2=1:length(S2_vec)
%         for ind_lr_inc=1:length(lr_inc_vec)
%             for ind_lr_dec=1:length(lr_dec_vec)
%                 for ind_er=1:length(er_vec)
%                     for ind_mc=1:length(mc_vec)
                        if ind_S2 > ind_S1
                            break;
                        end
                        net = feedforwardnet([S1_vec(ind_S1) S2_vec(ind_S2)]);
                        net.divideFcn = 'dividetrain';
                        net.trainFcn = 'traingdx';
                        net.performFcn = 'sse';
                        net.layers{1}.transferFcn='tansig';
                        net.layers{2}.transferFcn='tansig';
                        net.layers{3}.transferFcn='purelin';
                        net.trainParam.lr = 0.01; %
                        net.trainParam.epochs = 15000; %
                        net.trainParam.goal = 0.25;
%                         net.trainParam.lr_inc = lr_inc_vec(ind_lr_inc); %1.05;
%                         net.trainParam.lr_dec = lr_dec_vec(ind_lr_dec); % 0.7
%                         net.trainParam.max_perf_inc = er_vec(ind_er); % 1.04
%                         net.trainParam.mc = mc_vec(ind_mc); % 0.9
                        %net.trainParam.max_fail =200;
                        [net,tr] = train(net,Pn,Tn);
                        y = sim(net, Pn);
                        performance = perform(net,Tn,y);
                        sse
                        popr_klas = sum(abs(Tn-y)<=.5);
                        PK = popr_klas/length(Tn)*100;%, ind_lr_inc, ind_lr_dec, ind_er,ind_mc
                        PK_v6(ind_S1, ind_S2) = PK;
                        SSE_v6(ind_S1, ind_S2) = tr.perf(end);
                        epochs_v6(ind_S1, ind_S2) = tr.num_epochs;
                        fprintf("\nS1 = %d S2 = %d PK = %3.3f bestPerf= %d Epoka = %d",  S1_vec(ind_S1), S2_vec(ind_S2), PK, tr.best_perf, tr.num_epochs)
%                     end
%                 end
%             end
%         end
    end
end