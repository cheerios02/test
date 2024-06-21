% Load the optimal group assignments and necessary data for all simulations 
load('inputMCXG3_het_CO2.txt');
load('inputMCYG3_het_CO2.txt');
load('assign_G3_Het_C02.txt');
load('theta_G3_Het_new_CO2.txt');
load('BigG_perm_G3_het_CO2.mat')

% Define the number of periods, variables, replications, countries, and groups
T = 19;
Var = 8;
N = 21;
repNum = 50;
optGroup = BigG_perm;
G = 3;

% Initialize variables
thetas = zeros(repNum,G*Var);

% Initiate the simulations
for sim = 1:repNum
    disp(sim)
    X = inputMCXG3_het_CO2((sim-1)*N*T+1:sim*N*T,:); % Obtain the data for X for all countries and periods in the current simulation
    Y = inputMCYG3_het_CO2((sim-1)*N*T+1:sim*N*T); % Obtain the data for Y for all countries and periods in the current simulation
    opt_group_assign = optGroup(:,sim); % Obtain the optimal group assignment for all countries in the current simulation
    which_group=zeros(N,G);

    for g=1:G
        which_group(:,g)=(opt_group_assign==g); % Stores the optimal group of each country in the current simulation
    end

    Y_group_aver=zeros(N*T,1);
    X_group_aver=zeros(N*T,Var);

    country_sum_per_group=sum(which_group); % Obtain the sum of total countries in each group in the current simulation

    for i=1:N
        if country_sum_per_group(opt_group_assign(i))>1 % If the country is not the only one in the group
            for t=1:T
                Yt=Y(t:T:N*T);
                Y_group_aver((i-1)*T+t)=mean(Yt(opt_group_assign==opt_group_assign(i))); % For each period, obtain the average of the Y data for the countries that belong to the same group
                Xt=X(t:T:N*T,:);
                X_group_aver((i-1)*T+t,:)=mean(Xt(opt_group_assign==opt_group_assign(i),:));
            end
        else % If the country is the only one in the group
            for t=1:T
                Yt=Y(t:T:N*T);
                Y_group_aver((i-1)*T+t)=mean(Yt(opt_group_assign==opt_group_assign(i)));
                Xt=X(t:T:N*T,:);
                X_group_aver((i-1)*T+t,:)=Xt(opt_group_assign==opt_group_assign(i),:);
            end
        end
    end

    denominator=zeros(Var,Var, G);
    numerator=zeros(Var,G);
    theta_aux = zeros(Var,G);

    for g = 1:G
        for i=1:N
            if opt_group_assign(i) == g
                for t=1:T
                    denominator(:,:,g) = denominator(:,:,g)+(X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(X((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:));
                    numerator(:,g) = numerator(:,g) +(X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(Y((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:));
                end
            end
        end
        theta_aux(:,g) = denominator(:,:,g) \ numerator(:,g);
        thetas(sim,:) = reshape(theta_aux,1,Var*G); % Obtain the estimates of the model parameters for each simulation
    end
end

thetas = thetas(~any(isnan(thetas),2),:);
%% Bias:
%true_thetas = [1.0573,1.0437,1.0311;-0.282,0.2397,0.0848;3.0497,1.6145,-0.3242;-1.5472,-1.3994,-1.2407;-0.9248,-0.9176,-0.5975;-0.0038,-0.0024,0.0001;0.0013,0.0004,-0.00006;0.0083,0.0021,0.0096];
true_thetas = [0.8544,0.9519,0.9658;0.0847,0.0577,0.1254;-0.425,0.7681,0.5967;-1.1846,-1.3381,-1.35;-0.6426,-0.7447,-0.8035;-0.0023,-0.00026,-0.0035;0.00029,0.00017,0.00056;0.0065,0.0041,0.0051];
disp('The bias of the theta across all simulations:')
%mean_theta = reshape(mean(thetas), [Var,G]);
mean_theta = reshape(mean(theta_G3_Het_new_CO2), [Var,G]);
disp(mean_theta)
disp(abs(1 - mean_theta./true_thetas))

%% Standard Deviation
disp('The standard deviation for theta across all simulations is:')
std_theta = reshape(std(thetas), [Var,G]);
disp(std_theta)