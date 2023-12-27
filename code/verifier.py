import argparse
import yaml
from typing import Optional
import torch

from deeppoly import DeepPoly
from networks import get_network
from utils.loading import parse_spec

DEVICE = "cpu"


def analyze(
    net: torch.nn.Module,
    inputs: torch.Tensor,
    eps: float,
    true_label: int,
    modelname: Optional[str] = None,
) -> bool:
    # load config from file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)  # type: ignore
        config["modelname"] = modelname
    net.zero_grad()

    verifier = DeepPoly(model=net, true_label=true_label, input=inputs, config=config)  # type: ignore

    return verifier.verify(inputs, eps)


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label, str(args.net)):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
