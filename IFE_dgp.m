MC_sim = 500;
r = 3;
G = 3;
N = 21;
T = 19;
K = 8;
thetas_final = [0.3494, 0.7853, 0.0344; -0.2079, -0.7161, 0.3178; -4.1, -6.8155, -1.5799; 3.0267, 0.4184, -0.8535; -0.0044, -0.9041, -0.3529; -0.0021, 0.0043, -0.0097; 0.0012, 0.00002, 0.0011; -0.0193, 0.0082, 0.0328];
MC_IFE_thetas = zeros(r, K, MC_sim);
MC_assign_list = zeros(MC_sim,N);
thetas = zeros(K,G);

for j = 1:MC_sim
    disp(j)
    fm = zeros(N,T);
    final_F = 0.6*randn(T,r);
    final_Lambda = randn(N,r);
    x1 = rand(21, 19)';
    x2 = rand(21, 19)';
    x3 = rand(21, 19)';
    x4 = rand(21, 19)';
    x5 = rand(21, 19)';
    x6 = rand(21, 19)';
    x7 = rand(21, 19)';
    x8 = rand(21, 19)';
    X_lagged = [reshape(x1,T*N,1), reshape(x2,T*N,1), reshape(x3,T*N,1), reshape(x4,T*N,1), reshape(x5,T*N,1), reshape(x6,T*N,1), reshape(x7,T*N,1), reshape(x8,T*N,1), ];

    e = rand(19,19)';
    sigma = sqrt(0.3 + (0.1*x1.^2 + 0.1*x2.^2+ 0.1*x3.^2+ 0.1*x4.^2+ 0.1*x5.^2+ 0.1*x6.^2+ 0.1*x7.^2+ 0.1*x8.^2)/8);
    err = sigma'*e;

    for i = 1:N
        for t = 1:T
            fm(i,t) = x1(t,i)*thetas_final(1) + x2(t,i)*thetas_final(2) + x3(t,i)*thetas_final(3) + x4(t,i)*thetas_final(4) + x5(t,i)*thetas_final(5) + x6(t,i)*thetas_final(6) + x7(t,i)*thetas_final(7) + x8(t,i)*thetas_final(8) + final_Lambda(i,:) * final_F(t,:)' + err(i,t);
        end
    end
    Y0 = reshape(fm',N*T,1);
    clear Y_lagged
    Y_lagged = Y0;

    % Initial allocation is the true parameter values
    F_first_it = final_F;
    thetas_opt_first_it = thetas_final;
    group_alloc_first_it = randi(G, N, 1);
    obj_value_initial = 10^10; % Resets the objective value

    delta = 1;
    s = 0;
    while delta > 0 && s <= 500
        s = s + 1;
        % Step 1 of Algorithm 3: Compute thetas
        proj_matrix = eye(T) - F_first_it * F_first_it' / T; % Construct the projection matrix
        for g = 1:G
            var1 = 0;
            var2 = 0;
            countries_in_group = find(group_alloc_first_it == g);
            for i = countries_in_group'
                X_port = X_lagged((i-1)*T+1:i*T, :);
                Y_port = Y_lagged((i-1)*T+1:i*T);
                var1 = var1 + X_port' * proj_matrix * X_port;
                var2 = var2 + X_port' * proj_matrix * Y_port;
            end
            if rcond(var1) > eps
                thetas(:, g) = var1 \ var2;
            end
        end

        % Step 2 of Algorithm 3: Compute F's
        U = zeros(N*T, 1); % Initialize the residual vector
        for i = 1:N
            for g = 1:G
                if group_alloc_first_it(i) == g
                    U((i-1)*T+1:i*T) = Y_lagged((i-1)*T+1:i*T) - X_lagged((i-1)*T+1:i*T, :) * thetas(:, g);
                end
            end
        end
        residuals = reshape(U, T, N)';
        [V, D] = eig(residuals' * residuals); % This is the covariance matrix
        [~, eigenvalues] = sort(diag(D), 'descend');
        F_IFE = V(:, eigenvalues(1:r));

        % Step 3 of Algorithm 3: Computes λ's
        Lambda_IFE = residuals * F_IFE / T;

        % Step 4 of Algorithm 3: Compute the new group allocation
        new_group_alloc = zeros(N, 1);
        country_res = zeros(N, 1);
        for i = 1:N
            y_it = Y_lagged((i-1)*T+1:i*T);
            x_it = X_lagged((i-1)*T+1:i*T, :);
            min_ssr = inf;
            best_group = 1;
            for g = 1:G
                residuals = y_it - x_it * thetas(:, g) - (Lambda_IFE(i, :) * F_IFE')';
                ssr = sum(residuals .^ 2);
                if ssr < min_ssr
                    min_ssr = ssr;
                    best_group = g;
                    country_res(i) = ssr;
                end
            end
            new_group_alloc(i) = best_group;
        end

        % Checks for empty groups
        countries_per_group = histcounts(new_group_alloc, 1:G+1);
        for value = 1:G
            if countries_per_group(value) == 0
                [~, ii] = max(abs(country_res)); % Selects the country with the biggest squared difference between alpha and residuals
                new_group_alloc(ii) = value;
                countries_per_group(value) = 1;
                country_res(ii) = 0.0;
            end
        end

        % Update the objective function
        obj_value = 0;
        for value = 1:N
            for c = 1:G
                aux = (Lambda_IFE(i, :) * F_IFE')'; % Computes the interactive fixed effect
                for t = 1:T
                    if group_alloc_first_it(value) == c
                        obj_value = obj_value + (Y_lagged((value-1)*T+t) - X_lagged((value-1)*T+t, :) * thetas(:, g) - aux(t,1))^2;
                    end
                end
            end
        end
        delta = norm(thetas - thetas_opt_first_it, 'fro')^2 + norm(Lambda_IFE * F_IFE' - Lambda_first_it * F_first_it', 'fro')^2;

        Lambda_first_it = Lambda_IFE;
        F_first_it = F_IFE;
        thetas_opt_first_it = thetas;
        group_alloc_first_it = new_group_alloc;

        if obj_value < obj_value_initial
            thetas_final_2 = thetas_opt_first_it;
            obj_value_initial = obj_value;
            final_groups_2 = group_alloc_first_it;
            final_F_2 = F_IFE;
            final_Lambda_2 = Lambda_IFE;
        end
    end
    MC_IFE_thetas(:, :, j) = thetas_final_2';
    MC_assign_list(j,:) = final_groups_2;
end