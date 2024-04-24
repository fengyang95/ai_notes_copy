from model_arena import ModelArena
from model_arena.extensions import ChatGPTEvaluator

ma = ModelArena()

dataset = "demo"
model = "deepseek-coder-6.7b-instruct"
target_model = "gpt-3.5-turbo-1106"

# we do not need any ignore or transform
evaluator = ChatGPTEvaluator(
    ignore=None,
    transform=None,
    model="gpt-4-0613",
    show_progress=True,
)
# match
success_df, failed_df = ma.match(
    dataset=dataset,
    model=model,
    target_model=target_model,
    evaluator=evaluator,
    upload=False,
)
print(success_df)
print(failed_df)
