% Load the necessary datasets and set the number of countries and
% simulations
load('assign_G3_Het_C02_changed.txt');
BigG = assign_G3_Het_C02_changed';
load('theta_G3_Het_new_CO2_changed.txt');
N = 21;
repNum = 50;
G = 3;
K = 8;

opt_group_assign= [3 3 1 3 3 3 3 1 3 3 2 1 1 2 3 2 3 3 3 3 3]';

% Denote all possible permutations
diff_perm_num=6; % There are 3! possible permutations. We select a sufficiently high number so we are confident all permutations are included
permutations = [[1;2;3],[1;3;2],[2;1;3],[2;3;1],[3;2;1],[3;1;2]];
BigG_perm = zeros(N,repNum);

obj_value = zeros(diff_perm_num,1);
for i = 1:repNum
    % Store the optimal group allocation for the current simulation
    group_col = BigG(:,i);
    for j = 1:diff_perm_num
        % Reorder the group allocation according to the j-th permutation and calculate the squared error
        groups_reordered = permutations(group_col,j);
        obj_value(j,1) = sum((opt_group_assign - groups_reordered).^2);
    end
    % Obtain the relabelling of the groups with the smallest deviation for the current simulation and store it
    [min_error,min_error_pos] = min(obj_value);
    BigG_perm(:,i) = permutations(group_col,min_error_pos);
end

% Compute the misclassification probability
v = BigG_perm - kron(opt_group_assign,ones(1,repNum));
missclas_prob = 1 - mean(mean(v==0));
disp('The misclassification probability for 3 groups is:')
disp(missclas_prob)

true_thetas = [1.0573,1.0437,1.0311;-0.282,0.2397,0.0848;3.0497,1.6145,-0.3242;-1.5472,-1.3994,-1.2407;-0.9248,-0.9176,-0.5975;-0.0038,-0.0024,0.0008;0.0013,0.00049,-0.0001;0.0083,0.0021,0.010];
mean_theta = reshape(mean(theta_G3_Het_new_CO2_changed), [K,G]);
disp('The bias of the theta across all simulations:')
disp((mean_theta-true_thetas)./true_thetas*100);
 
disp('The standard deviation for theta across all simulations is:')
std_theta = reshape(std(theta_G3_Het_new_CO2_changed), [K,G]);
disp(std_theta)
