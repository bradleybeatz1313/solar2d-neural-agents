--- neural_net.lua
--- Lightweight feedforward neural network for Solar2D agents.
--- Supports arbitrary layer sizes, multiple activation functions,
--- and genetic algorithm-compatible weight serialization.
---
--- Designed for neuroevolution: agents evolve their networks
--- through selection, crossover, and mutation rather than backprop.
---
--- Usage:
---   local nn = require("ai.neural_net")
---   local brain = nn.create({8, 12, 6, 4})  -- 8 inputs, 2 hidden, 4 outputs
---   local outputs = nn.forward(brain, inputs)
---   local child = nn.crossover(parent_a, parent_b)
---   nn.mutate(child, 0.1, 0.3)

local M = {}

-- ============================================================
-- Activation Functions
-- ============================================================

M.activations = {
    sigmoid = function(x)
        return 1.0 / (1.0 + math.exp(-x))
    end,

    tanh = function(x)
        return math.tanh(x)
    end,

    relu = function(x)
        return math.max(0, x)
    end,

    leaky_relu = function(x)
        return x > 0 and x or (0.01 * x)
    end,

    softmax = function(values)
        local max_val = -math.huge
        for _, v in ipairs(values) do
            if v > max_val then max_val = v end
        end
        local sum = 0
        local result = {}
        for i, v in ipairs(values) do
            result[i] = math.exp(v - max_val)
            sum = sum + result[i]
        end
        for i = 1, #result do
            result[i] = result[i] / sum
        end
        return result
    end,
}

-- ============================================================
-- Network Creation
-- ============================================================

--- Create a new neural network with random weights.
--- @param layer_sizes table Array of layer sizes (e.g., {8, 12, 4})
--- @param activation string Activation function name (default: "tanh")
--- @return table Neural network instance
function M.create(layer_sizes, activation)
    local network = {
        layers = layer_sizes,
        activation = activation or "tanh",
        weights = {},
        biases = {},
        fitness = 0,
        generation = 0,
    }

    -- Initialize weights with Xavier initialization
    for i = 1, #layer_sizes - 1 do
        local fan_in = layer_sizes[i]
        local fan_out = layer_sizes[i + 1]
        local limit = math.sqrt(6.0 / (fan_in + fan_out))

        network.weights[i] = {}
        network.biases[i] = {}

        for j = 1, fan_out do
            network.weights[i][j] = {}
            network.biases[i][j] = (math.random() * 2 - 1) * 0.1

            for k = 1, fan_in do
                network.weights[i][j][k] = (math.random() * 2 - 1) * limit
            end
        end
    end

    return network
end

-- ============================================================
-- Forward Pass
-- ============================================================

--- Run a forward pass through the network.
--- @param network table Neural network instance
--- @param inputs table Array of input values
--- @return table Array of output values
function M.forward(network, inputs)
    assert(#inputs == network.layers[1],
        "Input size mismatch: expected " .. network.layers[1] .. ", got " .. #inputs)

    local activate = M.activations[network.activation]
    local current = inputs

    for layer = 1, #network.weights do
        local next_values = {}
        local is_output = (layer == #network.weights)

        for j = 1, #network.weights[layer] do
            local sum = network.biases[layer][j]
            for k = 1, #current do
                sum = sum + network.weights[layer][j][k] * current[k]
            end

            if is_output then
                -- Output layer: use tanh for bounded outputs
                next_values[j] = math.tanh(sum)
            else
                next_values[j] = activate(sum)
            end
        end

        current = next_values
    end

    return current
end

-- ============================================================
-- Genetic Operators
-- ============================================================

--- Create a child network via uniform crossover of two parents.
--- @param parent_a table First parent network
--- @param parent_b table Second parent network
--- @return table Child network
function M.crossover(parent_a, parent_b)
    assert(#parent_a.layers == #parent_b.layers, "Parents must have same topology")

    local child = M.create(parent_a.layers, parent_a.activation)

    for i = 1, #child.weights do
        for j = 1, #child.weights[i] do
            -- Inherit bias from random parent
            if math.random() < 0.5 then
                child.biases[i][j] = parent_a.biases[i][j]
            else
                child.biases[i][j] = parent_b.biases[i][j]
            end

            for k = 1, #child.weights[i][j] do
                if math.random() < 0.5 then
                    child.weights[i][j][k] = parent_a.weights[i][j][k]
                else
                    child.weights[i][j][k] = parent_b.weights[i][j][k]
                end
            end
        end
    end

    child.generation = math.max(parent_a.generation, parent_b.generation) + 1
    return child
end

--- Mutate network weights in-place.
--- @param network table Network to mutate
--- @param rate number Probability of mutating each weight [0-1]
--- @param strength number Standard deviation of gaussian noise
function M.mutate(network, rate, strength)
    rate = rate or 0.1
    strength = strength or 0.3

    for i = 1, #network.weights do
        for j = 1, #network.weights[i] do
            -- Mutate bias
            if math.random() < rate then
                network.biases[i][j] = network.biases[i][j] + M._gaussian() * strength
            end

            for k = 1, #network.weights[i][j] do
                if math.random() < rate then
                    network.weights[i][j][k] = network.weights[i][j][k] + M._gaussian() * strength
                end
            end
        end
    end
end

-- ============================================================
-- Serialization
-- ============================================================

--- Flatten all weights and biases to a single array.
--- @param network table Neural network
--- @return table Flat array of all parameters
function M.serialize(network)
    local params = {}
    for i = 1, #network.weights do
        for j = 1, #network.weights[i] do
            table.insert(params, network.biases[i][j])
            for k = 1, #network.weights[i][j] do
                table.insert(params, network.weights[i][j][k])
            end
        end
    end
    return params
end

--- Load weights from a flat parameter array.
--- @param network table Target network (must have matching topology)
--- @param params table Flat array of parameters
function M.deserialize(network, params)
    local idx = 1
    for i = 1, #network.weights do
        for j = 1, #network.weights[i] do
            network.biases[i][j] = params[idx]; idx = idx + 1
            for k = 1, #network.weights[i][j] do
                network.weights[i][j][k] = params[idx]; idx = idx + 1
            end
        end
    end
end

--- Count total trainable parameters.
--- @param network table Neural network
--- @return number Total weight + bias count
function M.param_count(network)
    local count = 0
    for i = 1, #network.weights do
        for j = 1, #network.weights[i] do
            count = count + 1  -- bias
            count = count + #network.weights[i][j]  -- weights
        end
    end
    return count
end

-- ============================================================
-- Utility
-- ============================================================

--- Box-Muller transform for gaussian random numbers.
function M._gaussian()
    local u1 = math.random()
    local u2 = math.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
end

--- Deep copy a network (for cloning top performers).
function M.clone(network)
    local copy = M.create(network.layers, network.activation)
    local params = M.serialize(network)
    M.deserialize(copy, params)
    copy.fitness = network.fitness
    copy.generation = network.generation
    return copy
end

return M


--- Returns the index of the output with the highest activation.
--- @param outputs table Array of output activations
--- @return number Index (1-based) of the argmax output
function M.argmax(outputs)
    local best_idx, best_val = 1, -math.huge
    for i, v in ipairs(outputs) do
        if v > best_val then best_val = v; best_idx = i end
    end
    return best_idx
end


--- Normalize a set of inputs to [-1, 1] given their expected range.
--- @param inputs table Raw input values
--- @param min_vals table Per-input minimum expected values
--- @param max_vals table Per-input maximum expected values
--- @return table Normalized inputs
function M.normalize(inputs, min_vals, max_vals)
    local result = {}
    for i = 1, #inputs do
        local range = max_vals[i] - min_vals[i]
        if range ~= 0 then
            result[i] = (inputs[i] - min_vals[i]) / range * 2 - 1
        else
            result[i] = 0
        end
    end
    return result
end


--- Returns the total number of trainable parameters in the network.
--- @param network table Neural network
--- @return number
function M.param_count(network)
    local count = 0
    for i = 1, #network.weights do
        for j = 1, #network.weights[i] do
            count = count + 1
            count = count + #network.weights[i][j]
        end
    end
    return count
end
