# Model Arena

## Installation

Install the package directly from codebase, using the `dev` branch.

```shell
pip install git+ssh://git@code.byted.org/smart-infra/ModelArena.git@dev
```

## Usage

### Initialization

```python
from sqlalchemy import create_engine
from model_arena import ModelArena

engine = create_engine("")
model_arena = ModelArena(engine)
```

### Datasets

```python
# show all datasets meta information
print(model_arena.datasets.meta)

# update a raw dataset
# information should be a json serialized string
#   json.dumps({"component 1": xxx, "component 2": xxx, ...})
#
df = pd.DataFrame(columns=["tag", "information"])
model_arena.datasets.raw_datasets.update(
    dataset="new_raw_dataset",
    df=df,
)

# update a dataset
# instruction template is used to combined the components defined
# inside information
model_arena.datasets.update(
    dataset="new_dataset_v1",
    records={
        "raw_dataset": "new_raw_dataset",
        "instruction_template": "Component 1: {component 1}, Component 2: {component 2}", 
    },
)

# generate a new dataset from previous dataset
model_arena.datasets.update(
    dataset="new_dataset_v2",
    records={
        "raw_dataset": "new_dataset",
        # support we do not need component 2 anymore
        "instruction_template": "Component 1: {component 1}", 
    }
)
```

### Models

```python
# show all models meta information
print(model_arena.models.meta)

# update a new model information
model_arena.models.update(
    model="new_model",
    records={
        # optional fields, however, high recommend to fill them
        "model_path": "/mnt/bn/....",
        "author": "test user",
        "remark": "a new agi model :)",
        # to get the correct prompt from dataset instruction
        # {instruction} should be appeared here
        "prompt_template": "You are an agi model, you have to deal with task: {instruction}",
    }
)
```


### Inferences

```python
# get history inferences
print(model_arena.get_inferences(models="new_model", datasets="all"))

# generate new inferences
# currently it is a dummy inference, you should do the inference on your own
# this will only generate a dataframe for you
df = model_arena.generate_inferences(models="new_model", datasets="new_dataset")

# update inferences
# you should add a new column `output` to the previou dataframe
def generate(prompt: str) -> str:
    ...

df["output"] = df["prompt"].apply(lambda x: generate(x))
model_arena.update_inferences(df)
```

### Matches

```python
# get history matches
print(model_arena.get_matches(models="all", datasets="new_dataset"))

# generate new matches
# two models to be competed over each questions inside datasets
# model_id_x with model_output_x will be scored as model_score_x
# model_id_y wiht model_output_y will be scored as model_score_y
df = model_arena.generate_matches(model="new_model", datasets="new_dataset")

# update matches
def score(output: str) -> int:
    # this process can also be done by human
    ...

df["model_score_x"] = df["model_output_x"].apply(lambda x: score(x))
df["model_score_y"] = df["model_output_y"].apply(lambda x: score(x))
model_arena.update_matches(df)
```

### Rating

```python
# get rating for a specific dataset
print(model_arena.get_rating(datasets="new_dataset"))
```

# TODOs

- [ ] Match pairing methods.
- [ ] Inference engine implementation.
- [ ] Dataset manipulation (base ops for grid search). 