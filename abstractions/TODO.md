(1) Look for variables related to reward, and their (recursive) dependencies... the more distant from the reward, presumably the less relevant, though that is obviously just a heuristic.

(2) Consider low-impact state variables / irrelevant variables.

(a) e.g., in the UAV domain (or at least some versions), we added a number of drone UAVs that are not under agent control and should have little or no impact on the agent state... this would ideally be evident from analyzing the reward dependencies.

(b) Consider variables that have minimal impact on transitions, e.g., for most variables we know their range if you look at the state-invariants, which almost always have inequalities on each state fluent to provide [min,max] values.  Assuming a monotonic response (not always true), you can measure a variable's impact by looking at the different between substituting the corner cases of the min and max values.  You can rank variables by this difference.  (To be honest, I don't think you have to look too much at the stochasticity here, just look at the expected transition.)