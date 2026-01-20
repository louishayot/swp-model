import argparse
import os
import sys
import warnings
from ast import literal_eval

import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.datasets.text import get_text_testloader
from swp.models.metrics import classic_errors
from swp.test.text import test
from swp.utils.datasets import (
    enrich_for_plotting,
    get_char_to_id,
    get_evaluation_dataset,
    get_phoneme_to_id,
    get_train_dataset,
)
from swp.utils.models import get_model, load_weights
from swp.utils.paths import get_evaluation_dir, get_figures_dir, get_weights_dir
from swp.utils.setup import backend_setup, seed_everything, set_device
from swp.viz.test import (
    plot_frequency_errors,
    plot_length_errors,
    plot_position_errors_smooth,
    regression_plots,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name string (must be Ut_....__g{N})",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        required=True,
        help="Training name string",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Test dataloader batch size",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--retest",
        action="store_true",
        help="Regenerate test results",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Tests also on the training set",
    )

    args = parser.parse_args()
    model_name = args.model_name
    train_name = args.train_name
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    include_stress = args.include_stress
    error_meter = classic_errors

    seed_everything()
    backend_setup()
    device = set_device()

    # Validate model name includes correct vocab sizes
    char_to_id = get_char_to_id()
    phoneme_to_id = get_phoneme_to_id(include_stress)
    char_vocab_size = len(char_to_id)
    phoneme_vocab_size = len(phoneme_to_id)

    if f"__g{char_vocab_size}" not in model_name:
        raise ValueError(
            f"Model name {model_name} does not match char_vocab_size={char_vocab_size}"
        )
    if f"_v{phoneme_vocab_size}_" not in model_name:
        raise ValueError(
            f"Model name {model_name} does not match phoneme_vocab_size={phoneme_vocab_size}"
        )

    weights_dir = get_weights_dir() / model_name / train_name

    if checkpoint is None:
        checkpoints = [f.stem.split(".")[-1] for f in weights_dir.glob("*.pth")]
    else:
        checkpoints = [checkpoint]

    for checkpoint in checkpoints:

        results_dir = (
            get_evaluation_dir() / f"{model_name}" / f"{train_name}" / f"{checkpoint}"
        )
        figures_dir = (
            get_figures_dir()
            / f"{model_name}"
            / f"{train_name}"
            / f"{checkpoint}"
            / "evaluation"
        )

        model = get_model(model_name)
        load_weights(
            model=model,
            model_name=model_name,
            train_name=train_name,
            checkpoint=checkpoint,
            device=device,
        )

        # Validate it's a text model
        if not model.is_text:
            raise ValueError(f"Model {model_name} is not a text model (Ut)")

        results_dir = results_dir / "control"
        figures_dir = figures_dir / "control"

        results_dir.mkdir(exist_ok=True, parents=True)
        figures_dir.mkdir(exist_ok=True, parents=True)

        ### TESTING ###

        if args.retest or not (results_dir / "evaluation.csv").exists():
            test_df = get_evaluation_dataset()
            test_loader = get_text_testloader(batch_size, include_stress)
            test_results, _ = test(
                model=model,
                device=device,
                test_df=test_df,
                test_loader=test_loader,
                include_stress=include_stress,
                error_meter=error_meter,
                verbose=args.verbose,
            )
            test_results = enrich_for_plotting(test_results, include_stress)
            test_results.to_csv(results_dir / "evaluation.csv")

        if args.train and (args.retest or not (results_dir / "train.csv").exists()):
            train_df = get_train_dataset()
            train_loader = get_text_testloader(batch_size, include_stress, train_df)
            train_results, train_error = test(
                model=model,
                device=device,
                test_df=train_df,
                test_loader=train_loader,
                include_stress=include_stress,
                error_meter=error_meter,
                verbose=args.verbose,
            )
            train_results.to_csv(results_dir / "train.csv")
            train_results = enrich_for_plotting(train_results, include_stress)

        ### PLOTTING ###

        converters = {
            "Phonemes": literal_eval,
            "No_Stress": literal_eval,
            "Prediction": literal_eval,
            "Error_Indices": literal_eval,
        }

        test_results = pd.read_csv(
            results_dir / "evaluation.csv", index_col=0, converters=converters
        )

        plot_length_errors(test_results, figures_dir)
        plot_frequency_errors(test_results, figures_dir)
        plot_position_errors_smooth(test_results, figures_dir)
        regression_plots(test_results, figures_dir, "real")
        regression_plots(test_results, figures_dir, "both")

        if args.verbose:
            print("-" * 60)
