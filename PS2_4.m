%% Problem 4

N=10;
M=20;
epoch=3000;

% Initialization
x=2*rand(2,10)-1;

%% Online Gradient Descent
E_g=zeros(1,epoch);
W=[0.1,0.1];
ita_g=1/12;

for i=1:epoch
    E_g(i)=0.5*(norm(W*x))^2;
    W=W-ita_g*W*(x*x');
end

%% Weight Perturbation
sigma=0.01;
E=zeros(1,epoch);
E_w=zeros(1,epoch);
W=[0.1,0.1];
ita_w=1/((M*N+2)*(N+2));

for i=1:epoch
    noise=randn(1,2).*sigma;
    E(i)=0.5*(norm(W*x))^2;
    E_w(i)=0.5*(norm((W+noise)*x))^2;
    W=W-ita_w/sigma^2*(E_w(i)-E(i))*noise;
end

%% Node Perturbation
E=zeros(1,epoch);
E_n=zeros(1,epoch);
W=[0.1,0.1];
ita_n=1/((M+2)*(N+2));

for i=1:epoch
    noise=randn(1,N).*sigma;
    E(i)=0.5*(norm(W*x))^2;
    E_n(i)=0.5*(norm(W*x+noise))^2;
    W=W-ita_n/sigma^2*(E_n(i)-E(i))*noise*x';
end


%%
figure, plot(1:200, E_g(1:200));
xlabel('epoch');
set(gca, 'YScale', 'log');
ylabel('log training error');

figure,plot(1:epoch, E_w);
set(gca, 'YScale', 'log');
xlabel('epoch');
ylabel('log training error');

figure, plot(1:1000, E_n(1:1000));
set(gca, 'YScale', 'log');
xlabel('epoch');
ylabel('log training error');
% legend('online gradient descent','weight perturbation','node perturbation','location','best');


%%
figure, plot(1:20, log(E_g(1:20)));
hold on;
plot(1:20, log(E_w(1:20)));
plot(1:20, log(E_n(1:20)));
xlabel('epoch');
ylabel('log training error');
legend('online gradient descent','weight perturbation','node perturbation','location','best');


%%
[k_g,~]=polyfit(1:200,log(E_g(1:200)),1);
t_g=-k_g(1)*N

[k_w,~]=polyfit(1:500,log(E_w(1:500)),1);
t_w=-k_w(1)*N

[k_n,~]=polyfit(1:70,log(E_n(1:70)),1);
t_n=-k_n(1)*N

t_g/t_w
t_n/t_w


%%
sigma=0.001:0.001:0.02;
t_g=zeros(1,length(sigma));
t_w=zeros(1,length(sigma));
t_n=zeros(1,length(sigma));

for k=1:length(sigma)
    E=zeros(1,epoch);
    E_w=zeros(1,epoch);
    W=rand(1,2);
    ita_w=1/((M*N+2)*(N+2));

    for i=1:epoch
        noise=randn(1,2).*sigma(k);
        E(i)=0.5*(norm(W*x))^2;
        E_w(i)=0.5*(norm((W+noise)*x))^2;
        W=W-ita_w/sigma(k)^2*(E_w(i)-E(i))*noise;
    end
    
    E=zeros(1,epoch);
    E_n=zeros(1,epoch);
    W=rand(1,2);
    ita_n=1/((M+2)*(N+2));

    for i=1:epoch
        noise=randn(1,N).*sigma(k);
        E(i)=0.5*(norm(W*x))^2;
        E_n(i)=0.5*(norm(W*x+noise))^2;
        W=W-ita_n/sigma(k)^2*(E_n(i)-E(i))*noise*x';
    end
%     
%     figure, plot(1:20, log(E_g(1:20)));
%     hold on;
%     plot(1:20, log(E_w(1:20)));
%     plot(1:20, log(E_n(1:20)));
%     xlabel('epoch');
%     ylabel('log training error');
%     legend('online gradient descent','weight perturbation','node perturbation','location','best');
    
    [k_g,~]=polyfit(1:10,log(E_g(1:10)),1);
    t_g(k)=-k_g(1)*N;

    [k_w,~]=polyfit(1:100,log(E_w(1:100)),1);
    t_w(k)=-k_w(1)*N;

    [k_n,~]=polyfit(1:50,log(E_n(1:50)),1);
    t_n(k)=-k_n(1)*N;
end
%%

figure,plot(1:k,t_g./t_w);
hold on
plot(1:k,t_n./t_w); 