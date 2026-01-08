--- evolution.lua
--- Genetic algorithm for evolving neural network populations.
--- Handles selection, crossover, mutation, and generational bookkeeping.
---
--- Supports tournament selection, elitism, and adaptive mutation rates.

local nn = require("ai.neural_net")

local M = {}

--- Create a new evolutionary population.
--- @param pop_size number Number of agents per generation
--- @param layer_sizes table Neural network topology
--- @param config table Optional GA parameters
--- @return table Population instance
function M.create_population(pop_size, layer_sizes, config)
    config = config or {}
    
    local pop = {
        size = pop_size,
        layer_sizes = layer_sizes,
        networks = {},
        generation = 0,
        best_fitness = -math.huge,
        avg_fitness = 0,
        history = {},
        
        -- GA parameters
        elitism = config.elitism or 0.1,           -- Top % preserved unchanged
        mutation_rate = config.mutation_rate or 0.15,
        mutation_strength = config.mutation_strength or 0.3,
        crossover_rate = config.crossover_rate or 0.7,
        tournament_size = config.tournament_size or 3,
        adaptive_mutation = config.adaptive_mutation ~= false,
    }
    
    -- Initialize random population
    for i = 1, pop_size do
        pop.networks[i] = nn.create(layer_sizes)
    end
    
    return pop
end

--- Evolve to the next generation based on fitness scores.
--- Call after all agents have been evaluated and their networks have fitness set.
--- @param pop table Population instance
--- @return table Updated population
function M.evolve(pop)
    -- Sort by fitness (descending)
    table.sort(pop.networks, function(a, b) return a.fitness > b.fitness end)
    
    -- Record stats
    local total_fitness = 0
    for _, net in ipairs(pop.networks) do
        total_fitness = total_fitness + net.fitness
    end
    
    pop.best_fitness = pop.networks[1].fitness
    pop.avg_fitness = total_fitness / pop.size
    pop.generation = pop.generation + 1
    
    table.insert(pop.history, {
        generation = pop.generation,
        best = pop.best_fitness,
        average = pop.avg_fitness,
        worst = pop.networks[pop.size].fitness,
    })
    
    -- Build next generation
    local next_gen = {}
    local elite_count = math.max(1, math.floor(pop.size * pop.elitism))
    
    -- 1. Elitism: copy top performers unchanged
    for i = 1, elite_count do
        local elite = nn.clone(pop.networks[i])
        elite.fitness = 0  -- Reset for next evaluation
        table.insert(next_gen, elite)
    end
    
    -- 2. Fill remaining slots with offspring
    while #next_gen < pop.size do
        local parent_a = M._tournament_select(pop)
        
        local child
        if math.random() < pop.crossover_rate then
            local parent_b = M._tournament_select(pop)
            child = nn.crossover(parent_a, parent_b)
        else
            child = nn.clone(parent_a)
        end
        
        -- Mutate
        local rate = pop.mutation_rate
        local strength = pop.mutation_strength
        
        if pop.adaptive_mutation then
            rate, strength = M._adaptive_mutation(pop, child)
        end
        
        nn.mutate(child, rate, strength)
        child.fitness = 0
        table.insert(next_gen, child)
    end
    
    pop.networks = next_gen
    return pop
end

--- Tournament selection: pick best from random subset.
--- @param pop table Population
--- @return table Selected network
function M._tournament_select(pop)
    local best = nil
    
    for _ = 1, pop.tournament_size do
        local idx = math.random(1, #pop.networks)
        local candidate = pop.networks[idx]
        if best == nil or candidate.fitness > best.fitness then
            best = candidate
        end
    end
    
    return best
end

--- Adaptive mutation: increase rates when population stagnates.
--- @param pop table Population
--- @param network table Current network being mutated
--- @return number, number Adjusted rate and strength
function M._adaptive_mutation(pop, network)
    local rate = pop.mutation_rate
    local strength = pop.mutation_strength
    
    -- Check if fitness has stagnated
    if #pop.history >= 5 then
        local recent = {}
        for i = math.max(1, #pop.history - 4), #pop.history do
            table.insert(recent, pop.history[i].best)
        end
        
        local improvement = recent[#recent] - recent[1]
        if math.abs(improvement) < 0.01 then
            -- Stagnation detected: increase mutation
            rate = math.min(0.5, rate * 2.0)
            strength = math.min(1.0, strength * 1.5)
        end
    end
    
    -- Lower-fitness networks get higher mutation (more exploration)
    local fitness_rank = network.fitness / math.max(1, pop.best_fitness)
    local rank_factor = 1.0 + (1.0 - fitness_rank) * 0.5
    
    return rate * rank_factor, strength * rank_factor
end

--- Get the best network from the current population.
--- @param pop table Population
--- @return table Best neural network
function M.get_best(pop)
    local best = pop.networks[1]
    for _, net in ipairs(pop.networks) do
        if net.fitness > best.fitness then
            best = net
        end
    end
    return best
end

--- Get stats string for display.
--- @param pop table Population
--- @return string Formatted stats
function M.stats_string(pop)
    return string.format(
        "Gen %d | Best: %.1f | Avg: %.1f | Pop: %d",
        pop.generation, pop.best_fitness, pop.avg_fitness, pop.size
    )
end

--- Save population to a serializable table.
--- @param pop table Population
--- @return table Serializable data
function M.save(pop)
    local data = {
        generation = pop.generation,
        layer_sizes = pop.layer_sizes,
        best_fitness = pop.best_fitness,
        history = pop.history,
        networks = {},
    }
    for i, net in ipairs(pop.networks) do
        data.networks[i] = {
            params = nn.serialize(net),
            fitness = net.fitness,
        }
    end
    return data
end

--- Load a population from saved data.
--- @param data table Previously saved data
--- @return table Restored population
function M.load(data)
    local pop = M.create_population(#data.networks, data.layer_sizes)
    pop.generation = data.generation
    pop.best_fitness = data.best_fitness
    pop.history = data.history
    
    for i, saved_net in ipairs(data.networks) do
        nn.deserialize(pop.networks[i], saved_net.params)
        pop.networks[i].fitness = saved_net.fitness
    end
    
    return pop
end

return M


--- Returns the top N networks from a population by fitness.
--- @param population table Array of network tables
--- @param n number Number of elites to select
--- @return table Array of top-n networks (sorted descending)
function M.top_n(population, n)
    local sorted = {}
    for _, net in ipairs(population) do
        table.insert(sorted, net)
    end
    table.sort(sorted, function(a, b) return a.fitness > b.fitness end)
    local elites = {}
    for i = 1, math.min(n, #sorted) do
        table.insert(elites, sorted[i])
    end
    return elites
end


--- Compute average fitness across a population.
--- @param population table Array of network tables
--- @return number Average fitness
function M.average_fitness(population)
    if #population == 0 then return 0 end
    local total = 0
    for _, net in ipairs(population) do total = total + net.fitness end
    return total / #population
end

--- Compute fitness standard deviation.
--- @param population table Array of network tables
--- @return number Standard deviation
function M.fitness_std(population)
    local avg = M.average_fitness(population)
    if #population < 2 then return 0 end
    local variance = 0
    for _, net in ipairs(population) do
        local diff = net.fitness - avg
        variance = variance + diff * diff
    end
    return math.sqrt(variance / (#population - 1))
end


--- Tournament selection: pick winner of k random agents by fitness.
--- @param population table Agent population
--- @param k number Tournament size (default 3)
--- @return table Selected network
function M.tournament_select(population, k)
    k = k or 3
    local best = nil
    for _ = 1, k do
        local candidate = population[math.random(#population)]
        if best == nil or candidate.fitness > best.fitness then
            best = candidate
        end
    end
    return best
end


--- Rank-based selection: assigns selection probability proportional to rank.
--- Less sensitive to fitness scale than roulette-wheel selection.
--- @param population table Agent population (sorted by fitness descending preferred)
--- @return table Selected network
function M.rank_select(population)
    local n = #population
    local sorted = {}
    for _, net in ipairs(population) do table.insert(sorted, net) end
    table.sort(sorted, function(a, b) return a.fitness > b.fitness end)
    -- Rank weights: rank 1 gets weight n, rank n gets weight 1
    local total = n * (n + 1) / 2
    local roll = math.random() * total
    local cumsum = 0
    for rank, net in ipairs(sorted) do
        cumsum = cumsum + (n - rank + 1)
        if roll <= cumsum then return net end
    end
    return sorted[#sorted]
end


--- Compute diversity score: average pairwise L2 distance between network weights.
--- High diversity = healthy population; low = premature convergence.
--- @param population table Array of networks
--- @param sample_size number Max pairs to sample (default 20 for performance)
--- @return number Average pairwise distance
function M.diversity_score(population, sample_size)
    local nn = require("ai.neural_net")
    sample_size = sample_size or 20
    local total, count = 0, 0
    for _ = 1, sample_size do
        local a = population[math.random(#population)]
        local b = population[math.random(#population)]
        local pa = nn.serialize(a)
        local pb = nn.serialize(b)
        local dist = 0
        for i = 1, #pa do dist = dist + (pa[i] - pb[i])^2 end
        total = total + math.sqrt(dist)
        count = count + 1
    end
    return count > 0 and total / count or 0
end
