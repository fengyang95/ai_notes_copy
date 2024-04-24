from loguru import logger

from notes.hparams.parser import get_all_args
from notes.data.data_load import load_all_dataset
from notes.model_and_tokenizer_loader.model_and_tokenizer_loader_family import load_model_and_tokenizer
from notes.context_builder.context_builder_family import get_context_builder
from notes.evaluate.evaluate import evaluate
from notes.metric.metric_score import get_acc_score
from notes.utils.set_seed import setup_seed


def run_eval(args=None):
    # Get eval args
    eval_args = get_all_args(args)
    logger.info('eval_args={}'.format(eval_args))

    # Setup seed
    setup_seed(eval_args.seed)

    # Get all dataset
    eval_datasets = load_all_dataset(eval_args)
    logger.info('Load all dataset success, total question number={}'.format(sum(len(v) for v in eval_datasets.values())))

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(eval_args)
    logger.info('Load model and tokenizer success')
    logger.info('tokenizer={}'.format(tokenizer))

    # load context_builder
    context_builder = get_context_builder(eval_args)
    logger.info('context_builder={}'.format(context_builder))

    # run model
    all_pred = evaluate(model, tokenizer, context_builder, eval_datasets)

    # get metric
    score_dict = get_acc_score(all_pred)
    logger.info('model_path={} k_shot={} Evaluation result={}'.format(eval_args.model_path, eval_args.k_shot, score_dict))

    # save metric


if __name__ == '__main__':
    run_eval()

