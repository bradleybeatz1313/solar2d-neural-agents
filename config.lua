-- config.lua

application = {
    content = {
        width = 1280,
        height = 720,
        scale = "letterBox",
        fps = 60,
        imageSuffix = {
            ["@2x"] = 2,
        },
    },
}

-- Training hyperparameters
Config.MUTATION_RATE     = 0.12   -- probability of mutating each weight
Config.MUTATION_STRENGTH = 0.25   -- gaussian noise standard deviation
Config.ELITISM_COUNT     = 4      -- top agents copied unchanged each generation
Config.CROSSOVER_PROB    = 0.6    -- probability of crossover vs. cloning

-- Network topology
Config.INPUT_SIZE  = 8   -- sensor inputs per agent
Config.HIDDEN_SIZES = {16, 12}  -- hidden layer sizes
Config.OUTPUT_SIZE = 4   -- action outputs

-- Display
Config.SHOW_SENSORS    = false   -- draw sensor rays on screen
Config.SHOW_FITNESS    = true    -- display fitness above each agent
Config.SHOW_GENERATION = true    -- show current generation counter

-- Fitness function weights
Config.FITNESS_SURVIVAL_WEIGHT  = 0.4
Config.FITNESS_DISTANCE_WEIGHT  = 0.4
Config.FITNESS_EFFICIENCY_WEIGHT = 0.2
