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
