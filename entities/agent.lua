--- agent.lua
--- Agent entity for the neural simulation. Manages visual representation,
--- physics state, and lifetime metrics.

local M = {}

function M.create(x, y, network)
    local visual = display.newCircle(x, y, 8)
    visual:setFillColor(0.3, 0.8, 0.9)
    visual.strokeWidth = 1
    visual:setStrokeColor(0.5, 0.5, 0.6)
    
    -- Direction indicator
    local indicator = display.newLine(x, y, x + 10, y)
    indicator:setStrokeColor(1, 1, 1, 0.5)
    indicator.strokeWidth = 2
    
    return {
        visual = visual,
        indicator = indicator,
        network = network,
        energy = 100,
        food_collected = 0,
        time_alive = 0,
        distance_traveled = 0,
    }
end

function M.destroy(agent)
    if agent.visual and agent.visual.removeSelf then
        agent.visual:removeSelf()
    end
    if agent.indicator and agent.indicator.removeSelf then
        agent.indicator:removeSelf()
    end
end

return M
