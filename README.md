# A torch native DeepPoly Verification Implementation (part of RTAI 2023 Course Project)



## Folder structure
In the directory `code`, you can find 3 files and one folder:
- The file `networks.py` contains the fully-connected and convolutional neural network architectures used in this project.
- The file `verifier.py` contains the of the verifier function. It implements loading of the stored networks and test cases. As well as the `analyze` function which executes a DeepPoly convex relaxation.
- The file `evaluate` will run all networks and specs currently in the repository. It follows the calling convention we will use for grading.
- The folder `utils` contains helper methods for loading and initialization
- In the directory `models`, you can find 14 neural networks (9 fully connected and 5 convolutional) weights. These networks are loaded using PyTorch in `verifier.py`. Note that we included two `_base` networks which do not contain activation functions.

- The folder `code/verify` contains all the Layer-Specific verifyer class (implemented as torch Modules) as well as the Abstract class Verifier, in case you wish to add any specific layers this should give you a good template to build upon. If your verifier has any optimizable values make sure to specify them as torch Parameters.
- In the file `code/deeppoly.py` you will find an example implementation of how to: Generate a Deeppoly Verfier from a pytorch network, how to optimize over verification parameters (using PGD with Adam) and how to run verification of samples.

In the directory `test_cases`, you can find 13 subfolders (the folder for `fc_6` contains both examples for `cifar10` and `mnist`). Each subfolder is associated with one of the networks using the same name. In a subfolder corresponding to a network, you can find 2 test cases for each network. Note that for the base networks, we provide you with 5 test cases each. Also, as we use 2 different versions (mnist, cifar10) of `fc_6`, the corresponding folder contains 2 test cases per dataset. As explained in the lecture, these test cases **are not** part of the set of test cases that we will use for the final evaluation.

Note that all inputs are images with pixel values between 0 and 1. The same range also applies to all abstract bounds that we want to verify.

## Setup instructions

We recommend you install a [Python virtual environment](https://docs.python.org/3/library/venv.html) to ensure dependencies are the same as the ones we will use for evaluation.
You can create a virtual environment and install the dependencies using the following commands:

```bash
$ virtualenv venv --python=python3.10
$ source venv/bin/activate
$ pip install -r requirements.txt
```

If you prefer conda environments we also provide a conda `environment.yaml` file which you can install (After installing [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation.html)) via

```bash
$ conda env create -f ./environment.yaml
$ conda activate deeppoly
```

We will run your verifier from `code` directory using the command:

```bash
$ python code/verifier.py --net {net} --spec test_cases/{net}/img{id}_{dataset}_{eps}.txt
```

In this command,
- `net` is equal to one of the following values (each representing one of the networks we want to verify): `fc_base, fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7, conv_base, conv_1, conv_2, conv_3, conv_4`.
- `id` is simply a numerical identifier of the case. They are not always ordered as they have been directly sampled from a larger set of cases.
- `dataset` is the dataset name, i.e.,  either `mnist` or `cifar10`.
- `eps` is the perturbation that the verifier should certify in this test case.

To test your verifier, you can run, for example:

```bash
$ python code/verifier.py --net fc_1 --spec test_cases/fc_1/img0_mnist_0.1394.txt
```

To evaluate the verifier on all networks and sample test cases, we provide an evaluation script.
You can run this script from the root directory using the following commands:

```bash
chmod +x code/evaluate
code/evaluate
```
