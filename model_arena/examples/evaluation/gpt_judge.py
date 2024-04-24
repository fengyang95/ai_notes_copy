from model_arena import ModelArena
from model_arena.extensions import ChatGPTEvaluator

ma = ModelArena()

dataset = "IDE164"
model = "deepseek-coder-6.7b-instruct"
target_model = "gpt-4-0613"

# we do not need any ignore or transform
evaluator = ChatGPTEvaluator(ignore=None, transform=None, model="gpt-4-0613", show_progress=True)
# match
df = ma.match(dataset=dataset, model=model, target_model=target_model, evaluator=evaluator, upload=False)
print(df)
