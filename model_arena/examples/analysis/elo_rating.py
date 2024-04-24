from model_arena import ModelArena
from model_arena.extensions import EloRatingAnalyst

ma = ModelArena()

analyst = EloRatingAnalyst(shift=1500, scale=400)
result = ma.analysis(datasets="IDE164", models="all", analyst=analyst)

print(result)
