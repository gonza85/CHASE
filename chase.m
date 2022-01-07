function chase(input_file)

parametros = table2array(readtable(input_file));
L=parametros(1);        %number of cities
b = parametros(2);      %birth rate, ususally 0
mu = parametros(3);     %death rate, usually 0
MaxTime = parametros(4); %length of simulation
nrea = parametros(5);   % number of independent runs
gamma = parametros(6);  % social tie parameter
alpha = parametros(7);  % distance exponent parameter
delta = parametros(8);  % max. fraction of people travelling each time step
K_all = parametros(9);  % optimal ocupation capacity for all cities
tau_beta = parametros(10);
tau_rho = parametros(11);

%rng(18458743) %seed used to generate the same map everytime
n0 = 3000*ones(L,1);  % same initial population for each city
n0 = diag(n0);
%x = 1e3*rand(L, 2);  % select positions of every city inside a 1000 by 1000 square.
% Map used in figure 1 of the paper
x = [256.36,844.41;
     532.32,610.81;
     329.68,94.231;
     583.63,468.62;
     76.742,537.12;
     922.31,545.79;
     344.95,673.96;
     555.89,624.4;
     348.4,157.72; 
     194.18,8.8633];
max_pop = K_all * ones(L,1);  % set same optimal population for every city


%% create folder and filename
folder = join(['2022-all-runs-N-3000-', ...
               'L', string(L), ...
               'MaxTime', string(MaxTime), ...
               'Nrea', string(nrea)], {'-'} );
mkdir(folder);
variables = strcat('gamma-', string(gamma), ...
                   '-delta-', string(delta), ...
                   '-tau_beta-', string(tau_beta), ...
                   '-tau_rho-', string(tau_rho), ...
                   '-K-', string(K_all), ...
                   '-alpha-', string(alpha));
filename = strcat(folder, '/', variables);



%%  Initialize other variables
tspan = (1:MaxTime+1)'; %time axis from 1 to MaxTime+1
tptos = length(tspan);  %number of points in the time axis
n_final = zeros(tptos, L);
%w = zeros(L,L,L); % w(j,i,k) prob. of people with origin k to move from i to j
w_final = zeros(L,L,tptos);
s_avrg = zeros(L,1);
surviving_cities = zeros(nrea,1);
diversity_vs_rea = zeros(L,nrea);

j=0;

%% Main loop
rng('shuffle'); %initialize pseudo random number generator with a random seed
n_temp = zeros(size(n0));
for rea = 1:nrea
    [~, d, d_mean]=initialize_map(L, x);
    t = 1;
    n = n0;
    w = update_matrix(n0, t);
    n_this_run=zeros(tptos,L);
    %first point, for t=1
    for i = 1:L
        n_this_run(t, i) = sum(n(i, :));
    end
    n_final(t, :) = n_final(t, :) + n_this_run(t,:);
    
    for tj = 1:tptos
        while t < tspan(tj)  
            for k = 1:L % dealing with on origin at a time
                w_k = w(:,:,k);
                n_k = n(:,k);
                migrants_k = mnrnd(n_k,w_k'); % migrants_k(i,j) went from i to j
                n_temp(:,k) = sum(migrants_k)';
            end
            n = n_temp;%, pause
            n_temp(:) = 0;
                    
            % Update the values of w(i,j,k)
            w = update_matrix(n, t);

            %time
            t = t + 1;
            for i = 1:L
                n_this_run(t, i) = sum(n(i, :));
            end
            n_final(t, :) = n_final(t, :) + n_this_run(t,:);
        end %t

        w_final(:, :, tj) = w_final(:, :, tj) + sum(w, 3);
    end %tj

    % Surviving cities:
    final_pop=n_this_run(end,:);
    surviving_cities(rea)=length(final_pop(final_pop>=1));
    
    dlmwrite(strcat(filename, '-temporal-',int2str(rea),'.csv'), n_this_run );
    diversity_vs_rea(:,rea)=calculate_diversity(n,L);
    plot_temporal(strcat(filename, '-temporal-',int2str(rea)),tspan(1:end-1), n_this_run(1:end-1,:), L, max_pop, variables); %temporal evolution of cities population
    
end %rea

dlmwrite(strcat(filename, 'surviving_cities_vs_K.csv'), [K_all, mean(surviving_cities), std(surviving_cities)])
dlmwrite(strcat(filename, 'diversity_vs_rea.csv'), diversity_vs_rea)
dlmwrite(strcat(filename, 'surviving_cities_ALL.csv'), surviving_cities)

%% finish averages
w_final = w_final / (nrea * L);
w_final(:, :, end)

n_final = n_final / nrea;
n_final(end, :)

s_avrg = s_avrg/nrea;

dlmwrite(strcat(filename, '-diversity.csv'), [gamma s_avrg'])

return

%% Other functions
function w_out = update_matrix(n_in,tt) %%%n,L,gamma
    w_out=zeros(L,L,L);
    Ntot = sum(n_in, 2); % row is city; column is origin
    rho_t = rho(tau_rho, tt);
    beta_t = beta(tau_beta,tt);
    for jj = 1:L
        for ii =1:L
            for kk= 1:L
                if ii~=jj
                    sum_term = sum(n_in(ii,:)) + (gamma - 1) * n_in(ii,kk);
                    w_out(ii,jj,kk) = sum_term^beta_t * exp(-rho_t * Ntot(ii) / max_pop(ii) ) / ((d(ii,jj)/d_mean(jj))^alpha);
                else
                    if (beta_t < 1e-18)
                        w_out(ii,jj,kk) = 1 - delta + delta * exp(-rho_t * Ntot(jj)/max_pop(jj));
                    else
                        w_out(ii,jj,kk) = 1 - delta + (delta * ( rho_t * exp(1) * Ntot(jj)/(beta_t * max_pop(jj)))^beta_t * (exp(-rho_t * Ntot(jj)/max_pop(jj) )));
                    end
                end
            end
        end
    end

    for kk = 1:L
        w_out(:,:,kk) = new_normalize(w_out(:,:,kk), L);
        %w_out(:,:,kk) = normalize_by_col(w_out(:,:,kk), L);
    end
    
end



end

function [x, d, d_mean]=initialize_map(L,x)
    %x = 1e3*rand(L, 2);  % positions of every patch inside a 1 by 1 square.
    d = zeros(L, L); % distance from i to j, largest possible distance is sqrt(2)
    for i = 1 : L
        for tj = 1 : L
            d(i,tj) = real(sqrt((x(i,1)-x(tj,1))^2 + (x(i,2)-x(tj,2))^2));
        end
    end
    d_mean = sum(d,1) / L;
end

function w_out = new_normalize(w_in, L)
w_out = zeros(L,L);
w_ii = diag(w_in)';
tmp = w_in - diag(w_ii);
suma = sum(tmp);
for j=1:L
    tmp(:,j) = tmp(:,j)*(1-w_ii(j))/suma(j);
end
w_out = tmp + diag(w_ii);
end

function h=plot_temporal(filename, t, n, L, Nmax, variables)
     figure('Name','Temporal', 'units', 'inch', 'position',[0,0,8,8]);
     pbaspect([1 1 1]);
     %fig.PaperOrientation = 'landscape';
     %fig.PaperType = 'uslegal';
     
     color=[224  31  31; %red
           245 139  15; %orange
           213 230  15; %lime yellow
            95 218  37; %green 
            24 175 175; %teal
            20 106 235; %blue
           137  46 209; %violet
           226  29 206; %pink
           100 100 100; %grey
             0   0   0]/255;   %black
    for i=1:L
        h = plot(t, n(:,i), 'Color', color(i,:),'LineWidth', 2);
        hold on
        %yyy = Nmax(i) * ones(size(t));
        %h = plot(t, yyy, '--', 'Color', color(i,:),'LineWidth', 1);
        %hold on
    end
    %legend(string([1;1]*(1:L)))
    legend(string(1:L));
    xlabel('Time','FontSize', 18);
    ylabel('Population','FontSize', 18);
    %ylim([0 9000]);
    hold off
    title(variables,'FontSize', 18);
    saveas(gcf, strcat(filename,'.fig'))
    %saveas(gcf, strcat(filename,'.png'))
    close
    return
end



function out = beta(tau_beta, t)

if tau_beta==0
    out = 1;
elseif tau_beta==999
    out=0;
else
    out = 1 - exp(-t / tau_beta);
end

end

function out = rho(tau_rho, t)

if tau_rho==0
    out = 1;
elseif tau_rho==999
    out=0;
else
    out = 1 - exp(-t / tau_rho);
end

end

function S = calculate_diversity(nn,L)
%% nn is a LxL matrix, where each row is the population of each city, and
% each element in that row is the number of people of a given origin.
% sum(nn,2) is the total population on each city
% sum(nn,1) is the total initial population on each city, that is constant over time. 

p = zeros(L,L);
tot = sum(nn,2);

for jj = 1:L
    if tot(jj)>0
        p(jj,:) = nn(jj,:)./tot(jj);
    end
end

S = ones(L,1) ./ sum(p .^ 2, 2);
S(S==Inf)=0;
% si p es cero, S es infinito. poner S=0 en ese caso
end
