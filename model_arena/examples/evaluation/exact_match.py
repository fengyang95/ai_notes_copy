import re

from model_arena import ModelArena
from model_arena.extensions import ExactMatchEvaluator

ma = ModelArena()

dataset = "niah_v1"
model = "deepseek-coder-6.7b-instruct"

# ignore output with <N/A>, which stands for
# the instruction template is longer than model context length
ignore = lambda x: x == "<N/A>"
# transfrom output, we hope that the language output is the last word possible
transform = lambda x: re.search(r"[\w]*", x.strip().split(" ")[-1])[0].lower()
# set up evaluator
evaluator = ExactMatchEvaluator(
    ignore=ignore,
    transform=transform,
    show_progress=False,
)
# evaluate
success_df, failed_df = ma.evaluate(
    dataset=dataset,
    model=model,
    evaluator=evaluator,
    upload=False,
)
print(success_df)
print(failed_df)
