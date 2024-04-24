from transformers import AutoTokenizer

from model_arena import ModelArena
from model_arena.extensions import NeedleInAHaystackAnalyst

ma = ModelArena()

dataset = "niah_v1"
model = "deepseek-coder-6.7b-instruct"
model_path = ma.models.get_model_path(model)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenize = lambda x: len(tokenizer.encode(x))

analyst = NeedleInAHaystackAnalyst(
    tokenize=tokenize,
    use_plot=True,
    plot_kwargs={"threshold": 0.9, "image": "niah_example.png"},
)
ma.analysis(datasets=dataset, models=model, analyst=analyst)
