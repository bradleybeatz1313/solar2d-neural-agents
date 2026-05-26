--- main.lua
--- Neural Agent Simulation — Solar 2D
--- Autonomous agents with neural network brains evolve via
--- neuroevolution to navigate, forage, and survive in a
--- procedurally generated environment.

local nn = require("ai.neural_net")
local evolution = require("ai.evolution")
local agent_mod = require("entities.agent")

-- ============================================================
-- Configuration
-- ============================================================

local CONFIG = {
    world_width = 1280,
    world_height = 720,
    
    pop_size = 30,
    network_topology = {8, 16, 8, 4},  -- 8 inputs, 2 hidden, 4 outputs
    
    generation_time = 20,     -- Seconds per generation
    food_count = 40,
    poison_count = 15,
    wall_count = 8,
    
    food_value = 10,
    poison_penalty = -20,
    survival_bonus = 0.5,     -- Per second alive
    movement_penalty = -0.01, -- Per unit moved (encourages efficiency)
}

-- ============================================================
-- State
-- ============================================================

local population
local agents = {}
local food_items = {}
local poison_items = {}
local walls = {}
local generation_timer = 0
local is_running = false
local speed_multiplier = 1

local ui_group
local stats_text

-- ============================================================
-- World Setup
-- ============================================================

local function spawn_food()
    for i = 1, CONFIG.food_count do
        local food = display.newCircle(
            math.random(40, CONFIG.world_width - 40),
            math.random(60, CONFIG.world_height - 40),
            6
        )
        food:setFillColor(0.2, 0.9, 0.3)
        food.is_consumed = false
        food_items[i] = food
    end
end

local function spawn_poison()
    for i = 1, CONFIG.poison_count do
        local poison = display.newCircle(
            math.random(40, CONFIG.world_width - 40),
            math.random(60, CONFIG.world_height - 40),
            5
        )
        poison:setFillColor(0.9, 0.2, 0.2)
        poison.is_consumed = false
        poison_items[i] = poison
    end
end

local function spawn_walls()
    for i = 1, CONFIG.wall_count do
        local w = math.random(60, 200)
        local h = math.random(10, 30)
        if math.random() > 0.5 then w, h = h, w end
        
        local wall = display.newRect(
            math.random(100, CONFIG.world_width - 100),
            math.random(100, CONFIG.world_height - 100),
            w, h
        )
        wall:setFillColor(0.4, 0.4, 0.5)
        walls[i] = wall
    end
end

local function clear_world()
    for _, f in ipairs(food_items) do
        if f and f.removeSelf then f:removeSelf() end
    end
    for _, p in ipairs(poison_items) do
        if p and p.removeSelf then p:removeSelf() end
    end
    food_items = {}
    poison_items = {}
end

-- ============================================================
-- Agent Neural Inputs
-- ============================================================

--- Build the 8-input sensor array for an agent.
--- Inputs: [nearest_food_dx, nearest_food_dy, nearest_food_dist,
---          nearest_poison_dx, nearest_poison_dy, nearest_poison_dist,
---          wall_ahead_dist, energy_ratio]
local function get_agent_inputs(agent)
    local ax, ay = agent.visual.x, agent.visual.y
    local inputs = {0, 0, 1, 0, 0, 1, 1, agent.energy / 100}
    
    -- Nearest food
    local min_food_dist = math.huge
    for _, food in ipairs(food_items) do
        if not food.is_consumed then
            local dx = food.x - ax
            local dy = food.y - ay
            local dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_food_dist then
                min_food_dist = dist
                inputs[1] = dx / CONFIG.world_width   -- Normalized dx
                inputs[2] = dy / CONFIG.world_height   -- Normalized dy
                inputs[3] = math.min(dist / 400, 1.0) -- Normalized distance
            end
        end
    end
    
    -- Nearest poison
    local min_poison_dist = math.huge
    for _, poison in ipairs(poison_items) do
        if not poison.is_consumed then
            local dx = poison.x - ax
            local dy = poison.y - ay
            local dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_poison_dist then
                min_poison_dist = dist
                inputs[4] = dx / CONFIG.world_width
                inputs[5] = dy / CONFIG.world_height
                inputs[6] = math.min(dist / 400, 1.0)
            end
        end
    end
    
    -- Wall proximity (simplified: distance to nearest boundary)
    local wall_dist = math.min(
        ax, ay,
        CONFIG.world_width - ax,
        CONFIG.world_height - ay
    )
    inputs[7] = math.min(wall_dist / 100, 1.0)
    
    -- Energy ratio
    inputs[8] = agent.energy / 100
    
    return inputs
end

-- ============================================================
-- Agent Neural Outputs → Actions
-- ============================================================

--- Map 4 neural outputs to agent actions.
--- Outputs: [move_x, move_y, boost, eat_radius]
local function apply_agent_outputs(agent, outputs, dt)
    local speed = 120 * dt * speed_multiplier
    
    -- Movement direction from outputs [1] and [2] (tanh range: -1 to 1)
    local dx = outputs[1] * speed
    local dy = outputs[2] * speed
    
    -- Boost: output [3] > 0 = sprint (costs more energy)
    if outputs[3] > 0 then
        dx = dx * 1.8
        dy = dy * 1.8
        agent.energy = agent.energy - 0.2 * dt * speed_multiplier
    end
    
    -- Move
    local new_x = math.max(10, math.min(CONFIG.world_width - 10, agent.visual.x + dx))
    local new_y = math.max(30, math.min(CONFIG.world_height - 10, agent.visual.y + dy))
    
    local actual_dist = math.sqrt((new_x - agent.visual.x)^2 + (new_y - agent.visual.y)^2)
    agent.distance_traveled = agent.distance_traveled + actual_dist
    
    agent.visual.x = new_x
    agent.visual.y = new_y
    
    -- Eat radius: output [4] controls how actively the agent forages
    local eat_radius = 15 + (outputs[4] + 1) * 10  -- Range: 5-25
    
    -- Check food collision
    for _, food in ipairs(food_items) do
        if not food.is_consumed then
            local fdx = food.x - agent.visual.x
            local fdy = food.y - agent.visual.y
            if math.sqrt(fdx*fdx + fdy*fdy) < eat_radius then
                food.is_consumed = true
                food.isVisible = false
                agent.energy = math.min(100, agent.energy + CONFIG.food_value)
                agent.food_collected = agent.food_collected + 1
            end
        end
    end
    
    -- Check poison collision
    for _, poison in ipairs(poison_items) do
        if not poison.is_consumed then
            local pdx = poison.x - agent.visual.x
            local pdy = poison.y - agent.visual.y
            if math.sqrt(pdx*pdx + pdy*pdy) < 12 then
                poison.is_consumed = true
                poison.isVisible = false
                agent.energy = agent.energy + CONFIG.poison_penalty
            end
        end
    end
    
    -- Passive energy drain
    agent.energy = agent.energy - 0.5 * dt * speed_multiplier
    
    -- Movement penalty (encourages efficient paths)
    agent.network.fitness = agent.network.fitness + CONFIG.movement_penalty * actual_dist
end

-- ============================================================
-- Simulation Loop
-- ============================================================

local function spawn_agents()
    for i, net in ipairs(population.networks) do
        agents[i] = agent_mod.create(
            math.random(100, CONFIG.world_width - 100),
            math.random(80, CONFIG.world_height - 80),
            net
        )
    end
end

local function cleanup_agents()
    for _, agent in ipairs(agents) do
        agent_mod.destroy(agent)
    end
    agents = {}
end

local function evaluate_fitness()
    for _, agent in ipairs(agents) do
        local net = agent.network
        net.fitness = net.fitness
            + agent.food_collected * CONFIG.food_value
            + agent.time_alive * CONFIG.survival_bonus
            + agent.energy * 0.1
    end
end

local function next_generation()
    evaluate_fitness()
    cleanup_agents()
    clear_world()
    
    evolution.evolve(population)
    
    spawn_food()
    spawn_poison()
    spawn_agents()
    generation_timer = 0
    
    update_stats_display()
end

local function game_loop(event)
    if not is_running then return end
    
    local dt = event.time / 1000 -- Not ideal but functional
    -- Use frame-based dt approximation
    dt = 1/60
    
    generation_timer = generation_timer + dt * speed_multiplier
    
    if generation_timer >= CONFIG.generation_time then
        next_generation()
        return
    end
    
    -- Update each agent
    for _, agent in ipairs(agents) do
        if agent.energy > 0 then
            local inputs = get_agent_inputs(agent)
            local outputs = nn.forward(agent.network, inputs)
            apply_agent_outputs(agent, outputs, dt)
            agent.time_alive = agent.time_alive + dt * speed_multiplier
            
            -- Visual feedback: color by energy
            local g = math.max(0.2, agent.energy / 100)
            agent.visual:setFillColor(0.3, g, 0.9)
        else
            agent.visual:setFillColor(0.3, 0.3, 0.3, 0.4)
        end
    end
end

-- ============================================================
-- UI
-- ============================================================

function update_stats_display()
    if stats_text then
        stats_text.text = evolution.stats_string(population)
    end
end

local function setup_ui()
    ui_group = display.newGroup()
    
    local bg = display.newRect(ui_group, CONFIG.world_width / 2, 15, CONFIG.world_width, 30)
    bg:setFillColor(0, 0, 0, 0.7)
    
    stats_text = display.newText(ui_group, "Initializing...", CONFIG.world_width / 2, 15, native.systemFont, 14)
    stats_text:setFillColor(1, 1, 1)
    
    -- Speed controls
    local speed_btn = display.newText(ui_group, "[1x]", CONFIG.world_width - 60, 15, native.systemFont, 14)
    speed_btn:setFillColor(0.5, 1, 0.5)
    speed_btn:addEventListener("tap", function()
        speed_multiplier = speed_multiplier == 1 and 5 or (speed_multiplier == 5 and 10 or 1)
        speed_btn.text = "[" .. speed_multiplier .. "x]"
    end)
end

-- ============================================================
-- Init
-- ============================================================

local function init()
    display.setDefault("background", 0.12, 0.12, 0.15)
    math.randomseed(os.time())
    
    population = evolution.create_population(
        CONFIG.pop_size,
        CONFIG.network_topology,
        {
            elitism = 0.1,
            mutation_rate = 0.15,
            mutation_strength = 0.3,
            adaptive_mutation = true,
        }
    )
    
    spawn_food()
    spawn_poison()
    spawn_walls()
    spawn_agents()
    setup_ui()
    
    is_running = true
    Runtime:addEventListener("enterFrame", game_loop)
    
    update_stats_display()
end

init()
