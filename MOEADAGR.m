classdef MOEADAGR < ALGORITHM
% <multi/many> <real/binary/permutation>
% Adaptive Replacement Strategies for MOEA/D (steady-state version)
% delta --- 0.8 --- The probability of selecting candidates from neighborhood
% type --- 2 --- The type of aggregation function
% Tm --- 0.1 --- The mating neighborhood size
% Trm --- 0.4 --- The maximum replacement neighborhood size
% nr --- 0.4 --- The maximum replacement number
% AS --- 1 --- The adaptive scheme
% 
% Author: Ruihao Zheng
% Last modified: 19/03/2024
% Ref: "Adaptive Replacement Strategies for MOEA/D"

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [delta, type, Tm, Trm, nr, AS] = Algorithm.ParameterSet(0.8, 2, 0.1, 0.4, 0.4, 1);

            %% initialization
            % Generate the weight vectors
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);
            Tm = ceil(Problem.N * Tm);
            Trm = ceil(Problem.N * Trm);
            nr = ceil(Problem.N * nr);
            % Detect the mating and replacement neighbors of each solution
            B = pdist2(W, W);
            [~,B] = sort(B, 2);
            Bm = B(:, 1:Tm);
            % Generate random population
            Population = Problem.Initialization();
            % Initialize the reference point
            z = min(Population.objs, [], 1);
            % Dertermine the scalar function
            switch type
                case 2
                    type = 2.1;
                    % W = 1./W ./ sum(1./W,2);  % seems to make no difference
                    % W = 1./W;
                otherwise
                    error('Unavailable value of p.')
            end
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Update the replacement neighborhood size
                Tr = updateTr(Trm, AS, Problem.FE, Problem.maxFE);
                Br = B(:, 1:Tr);

                % For each solution
                for i = 1 : Problem.N
                    % Choose the parents
                    if rand < delta
                        P = Bm(i, randperm(Tm));
                    else
                        P = randperm(Problem.N);
                    end

                    % Generate an offspring
                    Offspring = OperatorGAhalf(Population(P(1:2)));
                    
                    % Update the reference point
                    z = min(z, Offspring.obj);
                    
                    % find the most suitable problem for the offspring
                    g_O = calSubpFitness(type, Offspring.obj, z, W);
                    [~, Rj] = min(g_O);
                    
                    % Update the neighbors
                    R = Br(Rj,randperm(Tr));
                    g_old = calSubpFitness(type, Population(R).objs, z, W(R, :));
                    g_new = g_O(R);
                    Population(R(find(g_old>=g_new, nr))) = Offspring;
                end
            end
        end
    end
end


%%
function Tr = updateTr(Trm, AS, k, K, varargin)
% Update the parameter 'Tr' in AGRs
    
    keys = {'gamma', 'slope'};
    if ~isempty(varargin)
        isStr = find(cellfun(@ischar,varargin(1:end-1))&~cellfun(@isempty,varargin(2:end)));
        index = isStr(ismember(varargin(isStr),keys)) + 1;
        for i = 1 : length(isStr)
            str = varargin{isStr(i)};
            switch str
                case 'gamma'
                    gamma = varargin{index(i)};
                case 'slope'
                    slope = varargin{index(i)};
            end
        end
    end
    switch AS
        case 1
            % Sigmoid
            if ~exist('gamma', 'var')
                gamma = 0.25;  % center
            end
            if ~exist('slope', 'var')
                slope = 20;
            end
            Tr = ceil(Trm / (1 + exp(-slope * (k/K - gamma))));
        case 2
            % Linear
            Tr = ceil(k / K * Trm);
        case 3
            % Exponential
            if ~exist('slope', 'var')
                slope = 5;
            end
            Tr = ceil( ((exp(slope*k/K)-1) * Trm) / (exp(slope) - 1) );
        case 4
            % GR
            Tr = Trm;
    end
end


function g = calSubpFitness(type, objs, z, W)
% Calculate the function values of the scalarization method

    type2 = floor(type);
    switch type2
        case 2
            % Tchebycheff approach
            switch round((type - type2) * 10)
                case 1
                    g = max(abs(objs-z) ./ W, [], 2);
                otherwise
                    g = max(abs(objs-z) .* W, [], 2);
            end
    end
end
