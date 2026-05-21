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

-- Checkpointing
Config.SAVE_BEST_EVERY = 10   -- save best network every N generations
Config.CHECKPOINT_PATH = "checkpoints/"

-- Evolution strategy
Config.SELECTION_STRATEGY = "tournament"  -- "tournament" | "rank" | "roulette"
Config.TOURNAMENT_K = 3

-- Logging
Config.LOG_EVERY_N_GENS = 5   -- print stats every N generations
Config.LOG_DIVERSITY    = true  -- include diversity score in output

-- Adaptive mutation
Config.ADAPTIVE_MUTATION = true   -- scale mutation by diversity score
Config.MIN_MUTATION_STRENGTH = 0.05
Config.MAX_MUTATION_STRENGTH = 0.5

-- Speciation
Config.SPECIATION_ENABLED = false
Config.SPECIATION_THRESHOLD = 0.3
