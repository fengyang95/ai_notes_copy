from model_arena import ModelArena
from model_arena.extensions import PairwiseScoreStatAnalyst

ma = ModelArena()

analyst = PairwiseScoreStatAnalyst(pass_score=2.0, perfect_score=4.0)
result = ma.analysis(datasets="IDE164", models="gpt-4-0613", analyst=analyst)

print(result)
