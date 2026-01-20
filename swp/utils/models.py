from typing import TypedDict

import torch

from ..models.acoustic_encoder import AcousticEncoder
from ..models.autoencoder import Bimodel, Unimodel
from ..models.decoders import DecoderLSTM, DecoderRNN
from ..models.encoders import CorNetEncoder, EncoderLSTM, EncoderRNN
from .paths import get_weights_dir


def save_weights(
    model_name: str,
    train_name: str,
    model: Unimodel | Bimodel,
    epoch: int,
    checkpoint: int | None = None,
) -> None:
    r"""Save weights of a model for a given training procedure."""
    save_dir = get_weights_dir() / model_name / train_name
    save_dir.mkdir(exist_ok=True, parents=True)
    epoch_str = f"{epoch}"
    if checkpoint is not None:
        epoch_str = f"{epoch_str}_{checkpoint}"
    model_path = save_dir / f"{epoch_str}.pth"
    torch.save(model.state_dict(), model_path)


def load_weights(
    model: Unimodel | Bimodel,
    model_name: str,
    train_name: str,
    checkpoint: str,
    device: torch.device,
) -> None:
    r"""Load the weights of a model for a given training procedure at a specific
    epoch and potential checkpoint.
    """
    save_dir = get_weights_dir() / model_name / train_name
    model_path = save_dir / f"{checkpoint}.pth"
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.bind()


class CNNArgs(TypedDict):
    r"""TypedDict containing values required to create a visual encoder :
    `hidden_size` : hidden size of the network
    `cnn_model` : expected to contain values `"R"`, `"RT"`, `"S"` or `"Z"`.
    """

    hidden_size: int
    cnn_model: str


class AcousticArgs(TypedDict):
    r"""TypedDict containing values required to create an acoustic encoder :
    `input_dim` : input feature dimension (768 for wav2vec2-base, 1024 for large)
    """

    input_dim: int


class ModelArgs(TypedDict):
    r"""TypedDict containing values required to create a model :
    `model_class` : expected to contain values `"Ua"`, `"Ua_w2v"`, `"Uv"` or `"B"`
    `recur_type` : expected to contain values `"LSTM"` or `"RNN"`
    `hidden_size` : hidden size of the network
    `num_layers` : number of recurrent layers
    `vocab_size` : size of the vocabulary
    `droprate` : dropout ratio
    `tf_ratio` : teacher forcing ratio
    `start_token_id` : id of the token to use as first input for decoding
    `cnn_args` : `CNNArgs` dict containing the information for the visual decoder, or None if not relevant
    `acoustic_args` : `AcousticArgs` dict for acoustic encoder, or None if not relevant
    """

    model_class: str
    recur_type: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    droprate: float
    tf_ratio: float
    start_token_id: int
    cnn_args: CNNArgs | None
    acoustic_args: AcousticArgs | None


class TrainArgs(TypedDict):
    batch_size: int
    learning_rate: float
    fold_id: int | None
    include_stress: bool
    loss: str


def get_model_args(model_name: str) -> ModelArgs:
    r"""Create a dictionnary containing the necessary arguments to build a model
    from a `model_name`. See `ModelArgs` class for more information."""
    # Split on __ to separate main name from encoder-specific args
    big_split = model_name.split("__")
    main_name = big_split[0]
    name_split = main_name.split("_")

    # Check for Ua_w2v (acoustic) model
    if name_split[0] == "Ua" and len(name_split) > 1 and name_split[1] == "w2v":
        model_class = "Ua_w2v"
        recur_type = name_split[2]
        str_args = {arg[0]: arg[1:] for arg in name_split[3:]}
    else:
        model_class = name_split[0]
        recur_type = name_split[1]
        str_args = {arg[0]: arg[1:] for arg in name_split[2:]}

    cnn_args = None
    acoustic_args = None

    # Parse encoder-specific args from suffix
    if len(big_split) > 1:
        suffix = big_split[1]
        if suffix.startswith("c"):
            # CNN args for visual encoder
            cnn_str = suffix[1:]
            str_cnn_args = {arg[0]: arg[1:] for arg in cnn_str.split("_")}
            cnn_args = CNNArgs(
                {
                    "hidden_size": int(str_cnn_args["h"]),
                    "cnn_model": str_cnn_args["m"],
                }
            )
        elif suffix.startswith("g"):
            # Acoustic args (g for "generic" acoustic config)
            acoustic_str = suffix[1:]
            # Parse acoustic args - format: i{input_dim}
            str_acoustic_args = {arg[0]: arg[1:] for arg in acoustic_str.split("_") if arg}
            input_dim = int(str_acoustic_args.get("i", 768))  # Default to wav2vec2-base
            acoustic_args = AcousticArgs({"input_dim": input_dim})

    model_args = ModelArgs(
        {
            "model_class": model_class,
            "recur_type": recur_type,
            "hidden_size": int(str_args["h"]),
            "num_layers": int(str_args["l"]),
            "vocab_size": int(str_args["v"]),
            "droprate": float(str_args["d"]),
            "tf_ratio": float(str_args["t"]),
            "start_token_id": int(str_args["s"]),
            "cnn_args": cnn_args,
            "acoustic_args": acoustic_args,
        }
    )
    return model_args


def get_model(model_name: str) -> Unimodel | Bimodel:
    r"""Create a model corresponding to the `model_name`"""
    model_args = get_model_args(model_name)
    recur_type = model_args["recur_type"].upper()
    if recur_type == "LSTM":
        audit_encoder_class = EncoderLSTM
        decoder_class = DecoderLSTM
    elif recur_type == "RNN":
        audit_encoder_class = EncoderRNN
        decoder_class = DecoderRNN
    else:
        raise NotImplementedError(
            f"Recurrent type {recur_type} is not currently supported"
        )
    decoder = decoder_class(
        vocab_size=model_args["vocab_size"],
        hidden_size=model_args["hidden_size"],
        num_layers=model_args["num_layers"],
        dropout=model_args["droprate"],
        tf_ratio=model_args["tf_ratio"],
    )
    model_class = model_args["model_class"]

    # Handle Ua_w2v (acoustic) models
    if model_class == "Ua_w2v":
        # Default to wav2vec2-base dimensions if not specified
        if model_args["acoustic_args"] is not None:
            input_dim = model_args["acoustic_args"]["input_dim"]
        else:
            input_dim = 768  # wav2vec2-base default
        encoder = AcousticEncoder(
            input_dim=input_dim,
            hidden_size=model_args["hidden_size"],
            num_layers=model_args["num_layers"],
            dropout=model_args["droprate"],
            recur_type=recur_type,
        )
        model = Unimodel(
            encoder=encoder,
            decoder=decoder,
            start_token_id=model_args["start_token_id"],
        )
    elif model_class.startswith("U"):
        if model_class == "Ua":
            encoder = audit_encoder_class(
                vocab_size=model_args["vocab_size"],
                hidden_size=model_args["hidden_size"],
                num_layers=model_args["num_layers"],
                dropout=model_args["droprate"],
            )
        elif model_class == "Uv":
            if model_args["cnn_args"] is None:
                raise ValueError(
                    "No arguments corresponding to the visual encoder in a visual model"
                )
            encoder = CorNetEncoder(
                hidden_size=model_args["cnn_args"]["hidden_size"],
                cornet_model=model_args["cnn_args"]["cnn_model"],
            )
        else:
            raise ValueError(
                f"Trying to name a Unimodel that is neither auditory, visual, nor acoustic: {model_class}"
            )
        model = Unimodel(
            encoder=encoder,
            decoder=decoder,
            start_token_id=model_args["start_token_id"],
        )
    elif model_class == "B":
        audit_encoder = audit_encoder_class(
            vocab_size=model_args["vocab_size"],
            hidden_size=model_args["hidden_size"],
            num_layers=model_args["num_layers"],
            dropout=model_args["droprate"],
        )
        if model_args["cnn_args"] is None:
            raise ValueError(
                "No arguments corresponding to the visual encoder in a visual model"
            )
        visual_encoder = CorNetEncoder(
            hidden_size=model_args["cnn_args"]["hidden_size"],
            cornet_model=model_args["cnn_args"]["cnn_model"],
        )
        model = Bimodel(
            audit_encoder=audit_encoder,
            visual_encoder=visual_encoder,
            decoder=decoder,
            start_token_id=model_args["start_token_id"],
        )
    else:
        raise ValueError(f"Model class not recognized : {model_class}")
    return model


def get_model_name(model: Unimodel | Bimodel) -> str:
    r"""Returns the codified `model_name` corresponding to the `model`"""
    suffix_str = None
    if isinstance(model, Unimodel):
        if model.is_auditory:
            model_name = "Ua"
        elif model.is_acoustic:
            model_name = "Ua_w2v"
            # Add acoustic args suffix
            suffix_str = f"i{model.encoder.input_dim}"
        elif model.is_visual:
            model_name = "Uv"
            suffix_str = f"h{model.encoder.hidden_size}_m{model.encoder.cnn_model}"
        else:
            model_name = "U_unknown"
    else:
        model_name = "B"
        suffix_str = (
            f"h{model.visual_encoder.hidden_size}_m{model.visual_encoder.cnn_model}"
        )
    if isinstance(model.decoder, DecoderLSTM):
        model_name = f"{model_name}_LSTM"
    elif isinstance(model.decoder, DecoderRNN):
        model_name = f"{model_name}_RNN"
    model_name = f"{model_name}_h{model.decoder.hidden_size}"
    model_name = f"{model_name}_l{model.decoder.num_layers}"
    model_name = f"{model_name}_v{model.decoder.vocab_size}"
    model_name = f"{model_name}_d{model.decoder.droprate}"
    model_name = f"{model_name}_t{model.decoder.tf_ratio}"
    model_name = f"{model_name}_s{model.start_token_id}"
    if suffix_str is not None:
        # Use 'c' prefix for CNN args (visual), 'g' prefix for acoustic args
        if model.is_acoustic if isinstance(model, Unimodel) else False:
            model_name = f"{model_name}__g{suffix_str}"
        else:
            model_name = f"{model_name}__c{suffix_str}"
    return model_name


def get_model_name_from_args(
    model_class: str,
    recur_type: str,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    droprate: float,
    tf_ratio: float,
    start_token_id: int,
    cnn_args: CNNArgs | None = None,
    acoustic_args: AcousticArgs | None = None,
    **kwargs,
) -> str:
    r"""Generate the `model_name` from the arguments that would allow to generate the model"""
    model_name = f"{model_class}_{recur_type.upper()}"
    model_name = f"{model_name}_h{hidden_size}"
    model_name = f"{model_name}_l{num_layers}"
    model_name = f"{model_name}_v{vocab_size}"
    model_name = f"{model_name}_d{droprate}"
    model_name = f"{model_name}_t{tf_ratio}"
    model_name = f"{model_name}_s{start_token_id}"
    if cnn_args is not None:
        cnn_str = f'h{cnn_args["hidden_size"]}_m{cnn_args["cnn_model"]}'
        model_name = f"{model_name}__c{cnn_str}"
    elif acoustic_args is not None:
        acoustic_str = f'i{acoustic_args["input_dim"]}'
        model_name = f"{model_name}__g{acoustic_str}"
    return model_name


def get_train_name(
    batch_size: int,
    learning_rate: float,
    fold_id: int | None,
    include_stress: bool,
    seed: int,
    loss: str = "classic",
    **kwargs,
) -> str:
    r"""Generate the `train_name` from the training arguments."""
    fold_str = "all" if fold_id is None else fold_id
    train_name = f"b{batch_size}_l{learning_rate}_f{fold_str}_s{seed}"
    if include_stress:
        train_name = f"{train_name}_sw"
    else:
        train_name = f"{train_name}_sn"

    if loss == "classic":
        train_name = f"{train_name}_ec"
    elif loss == "first":
        train_name = f"{train_name}_ef"

    # TODO add support for visual dataset, mixed or not
    return train_name


def get_train_args(train_name: str) -> TrainArgs:
    r"""Returns a dictionnary containing the arguments corresponding to the `train_name`."""
    # TODO add support for visual dataset, mixed or not
    str_args = {arg[0]: arg[1:] for arg in train_name.split("_")}
    if str_args["s"] == "w":  # include_stress
        include_stress = True
    elif str_args["s"] == "n":
        include_stress = False
    else:
        raise ValueError(f'Stress value not recognized : {str_args["s"]}')
    train_args = TrainArgs(
        {
            "batch_size": int(str_args["b"]),
            "learning_rate": float(str_args["l"]),
            "fold_id": None if str_args["f"] == "all" else int(str_args["f"]),
            "include_stress": include_stress,
            "loss": str_args["e"],
        }
    )
    return train_args
