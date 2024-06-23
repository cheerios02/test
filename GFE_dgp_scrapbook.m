MC_sim = 100;
G = 3;
N = 21;
T = 19;
K = 8;
thetas_final = [0.3494, 0.7853, 0.0344; -0.2079, -0.7161, 0.3178; -4.1, -2.8155, -1.5799; 3.0267, 0.4184, -0.8535; -0.44, -0.9041, -0.3529; -0.21, 1.043, -0.97; 0.12, 0.2, 0.11; -0.193, 0.82, 0.328];
MC_GFE_thetas = zeros(G, K, MC_sim);
MC_assign_list = zeros(MC_sim,N);
miss_prob_list = zeros(100,1);
av_list_thetas = zeros(G,K,100);

for times = 1:100
    disp(times)
    group_alloc_first_it = randi(G, N, 1);
    initial_group = group_alloc_first_it;
    fm = zeros(N,T);
    for j = 1:MC_sim
        delta_hat = rand(N,T);
        x1 = rand(21, 19)';
        x2 = rand(21, 19)';
        x3 = rand(21, 19)';
        x4 = rand(21, 19)';
        x5 = rand(21, 19)';
        x6 = rand(21, 19)';
        x7 = rand(21, 19)';
        x8 = rand(21, 19)';
        X_lagged = [reshape(x1,T*N,1), reshape(x2,T*N,1), reshape(x3,T*N,1), reshape(x4,T*N,1), reshape(x5,T*N,1), reshape(x6,T*N,1), reshape(x7,T*N,1), reshape(x8,T*N,1)];
        err = randn(21,19);

        for i = 1:N
            g = group_alloc_first_it(i);
            for t = 1:T
                fm(i,t) = x1(t,i)*thetas_final(1,g) + x2(t,i)*thetas_final(2,g) + x3(t,i)*thetas_final(3,g) + x4(t,i)*thetas_final(4,g) + x5(t,i)*thetas_final(5,g) + x6(t,i)*thetas_final(6,g) + x7(t,i)*thetas_final(7,g) + x8(t,i)*thetas_final(8,g) + delta_hat(i,t) + err(i,t);
            end
        end
        Y0 = reshape(fm',N*T,1);
        clear Y_lagged
        Y_lagged = Y0;

        W = Y_lagged - X_lagged * thetas_final(:,1);

        % Select G random centers
        V = randi(N, G, 1); % Selects G random countries from the set
        alphas_intermediate = zeros(T, G);
        for value = 1:G
            alphas_intermediate(:, value) = W((V(value)-1)*T+1:V(value)*T); % Selects the alpha values for all periods of these G countries
        end

        obj_value_initial = 10^10;
        delta = 1;
        s = 0;
        thetas_aux = thetas_final;

        while delta > 0 && s <= 100
            s = s+1;
            % Step 1: Assignment
            group_class_intermediate = zeros(N, G);
            for country = 1:N
                y = Y_lagged((country-1)*T+1:country*T); % Selects the data related to the dependent variable for each period of each country
                x = X_lagged((country-1)*T+1:country*T, :); % Selects the data related to the independent variables for each period of each country
                for group = 1:G
                    u = 0.0;
                    for period = 1:T
                        u = u + (y(period) - x(period,:) * thetas_aux(:, group) - alphas_intermediate(period, group))^2; % Step 2 of Algorithm 1
                    end
                    group_class_intermediate(country, group) = u;
                end
            end

            % Group classification
            [group_class, group_assign_intermediate] = min(group_class_intermediate, [], 2);
            countries_per_group = histcounts(group_assign_intermediate, 1:G+1);

            % Checks for empty groups as per Hansen and Mladenovic
            for value = 1:G
                if countries_per_group(value) == 0
                    [~, ii] = max(abs(group_class)); % Selects the country with the biggest squared difference between alpha and residuals
                    group_assign_intermediate(ii) = value;
                    countries_per_group(value) = 1;
                    group_class(ii) = 0.0;
                end
            end

            % Step 2: Update
            x1gt = zeros(T, G);
            x2gt = zeros(T, G);
            x3gt = zeros(T, G);
            x4gt = zeros(T, G);
            x5gt = zeros(T, G);
            x6gt = zeros(T, G);
            x7gt = zeros(T, G);
            x8gt = zeros(T, G);
            ygt = zeros(T, G);

            for value = 1:N
                for c = 1:G
                    if group_assign_intermediate(value) == c
                        for t = 1:T
                            x1gt(t, c) = x1gt(t, c) + X_lagged((value-1)*T+t, 1) / countries_per_group(c); % Computes the within-group mean of covariate 1 for each time period
                            x2gt(t, c) = x2gt(t, c) + X_lagged((value-1)*T+t, 2) / countries_per_group(c); % Computes the within-group mean of covariate 2 for each time period
                            x3gt(t, c) = x3gt(t, c) + X_lagged((value-1)*T+t, 3) / countries_per_group(c); % Computes the within-group mean of covariate 3 for each time period
                            x4gt(t, c) = x4gt(t, c) + X_lagged((value-1)*T+t, 4) / countries_per_group(c); % Computes the within-group mean of covariate 4 for each time period
                            x5gt(t, c) = x5gt(t, c) + X_lagged((value-1)*T+t, 5) / countries_per_group(c); % Computes the within-group mean of covariate 5 for each time period
                            x6gt(t, c) = x6gt(t, c) + X_lagged((value-1)*T+t, 6) / countries_per_group(c); % Computes the within-group mean of covariate 6 for each time period
                            x7gt(t, c) = x7gt(t, c) + X_lagged((value-1)*T+t, 7) / countries_per_group(c); % Computes the within-group mean of covariate 7 for each time period
                            x8gt(t, c) = x8gt(t, c) + X_lagged((value-1)*T+t, 8) / countries_per_group(c); % Computes the within-group mean of covariate 8 for each time period
                            ygt(t, c) = ygt(t, c) + Y_lagged((value-1)*T+t) / countries_per_group(c); % Computes the within-group mean of the response variable for each time period
                        end
                    end
                end
            end

            % Compute demeaned vectors
            X_demeaned = zeros(N*T, K);
            Y_demeaned = zeros(N*T, 1);
            thetas_new = zeros(K, G);
            for c = 1:G
                for value = 1:N
                    if group_assign_intermediate(value) == c
                        for t = 1:T
                            X_demeaned((value-1)*T+t, 1) = X_lagged((value-1)*T+t, 1) - x1gt(t, c); % Demeans the first covariate
                            X_demeaned((value-1)*T+t, 2) = X_lagged((value-1)*T+t, 2) - x2gt(t, c); % Demeans the second covariate
                            X_demeaned((value-1)*T+t, 3) = X_lagged((value-1)*T+t, 3) - x3gt(t, c); % Demeans the third covariate
                            X_demeaned((value-1)*T+t, 4) = X_lagged((value-1)*T+t, 4) - x4gt(t, c); % Demeans the fourth covariate
                            X_demeaned((value-1)*T+t, 5) = X_lagged((value-1)*T+t, 5) - x5gt(t, c); % Demeans the fifth covariate
                            X_demeaned((value-1)*T+t, 6) = X_lagged((value-1)*T+t, 6) - x6gt(t, c); % Demeans the sixth covariate
                            X_demeaned((value-1)*T+t, 7) = X_lagged((value-1)*T+t, 7) - x7gt(t, c); % Demeans the seventh covariate
                            X_demeaned((value-1)*T+t, 8) = X_lagged((value-1)*T+t, 8) - x8gt(t, c); % Demeans the eighth covariate
                            Y_demeaned((value-1)*T+t) = Y_lagged((value-1)*T+t) - ygt(t, c); % Demeans the response variable
                        end
                    end
                end
                theta_c = (X_demeaned' * X_demeaned) \ (X_demeaned' * Y_demeaned); % Computes the thetas using an OLS regression, Step 3 of Algorithm 1
                thetas_new(:, c) = theta_c;
            end

            % Update the objective function
            obj_value = 0;
            for value = 1:N
                for c = 1:G
                    if group_assign_intermediate(value) == c
                        for t = 1:T
                            obj_value = obj_value + (Y_lagged((value-1)*T+t) - X_lagged((value-1)*T+t, :) * thetas_new(:, c) - alphas_intermediate(t, c))^2;
                        end
                    end
                end
            end

            % Computes time-trends
            alphas = zeros(T, G);
            for value = 1:N
                for c = 1:G
                    if group_assign_intermediate(value) == c
                        for t = 1:T
                            alphas(t, c) = alphas(t, c) + (Y_lagged((value-1)*T+t) - X_lagged((value-1)*T+t, 1) * thetas_new(1, c) - X_lagged((value-1)*T+t, 2) * thetas_new(2, c) - X_lagged((value-1)*T+t, 3) * thetas_new(3, c) - X_lagged((value-1)*T+t, 4) * thetas_new(4, c) - X_lagged((value-1)*T+t, 5) * thetas_new(5, c) - X_lagged((value-1)*T+t, 6) * thetas_new(6, c) - X_lagged((value-1)*T+t, 7) * thetas_new(7, c) - X_lagged((value-1)*T+t, 8) * thetas_new(8, c)) / countries_per_group(c); % Step 3 of Algorithm 1
                        end
                    end
                end
            end

            delta = sum((thetas_new(:) - thetas_aux(:)).^2) + sum((alphas(:) - alphas_intermediate(:)).^2); % Necessary to test the convergence of the algorithm
            thetas_aux = thetas_new;
            alphas_intermediate = alphas;

            % Stores the optimal group assignments, theta values, time trends, etc.
            if obj_value < obj_value_initial
                thetas_opt = thetas_aux;
                alphas_opt = alphas;
                obj_value_initial = obj_value;
                opt_group_assign = group_assign_intermediate;
                countries_per_group_opt = countries_per_group;
            end
        end
         MC_GFE_thetas(:, :, j) = thetas_opt';
    end
    av_list_thetas(:,:,times) = mean(MC_GFE_thetas,3);
end