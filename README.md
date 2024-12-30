# R-MABM

The _Rational-Macro Agent Based Model_.
A Python (multi-agent) reinforcement learning interface of the "CATS" model ([_Emergent dynamics of a macroeconomic
agent based model with capital and credit_](https://www.sciencedirect.com/science/article/abs/pii/S0165188914001572)).

## Installation

To download and install the package, run:

```bash
git clone git@github.com:Brusa99/R-MABM.git
cd R-MABM
python -m pip install -e .
```

Otherwise, it can be installed directly with `pip install git+https://github.com/Brusa99/R-MABM.git`.

### Julia Dependencies

Julia is required to run the model.
If not present on your system, [juliacall](https://github.com/JuliaPy/PythonCall.jl) will install it automatically.

Moreover, the julia package [ABCredit](https://github.com/bancaditalia/ABCredit.jl) is required.
The package will be installed automatically when instantiating a `Cats` object.
Otherwise, you can install it manually by running in _Julia_ the following commands:

```julia
using Pkg
Pkg.add("ABCredit")
```

## Disclaimer

This package is an outcome of a research project.
The article was presented at [ICAIF24](https://ai-finance.org/), and is available [here](https://dl.acm.org/doi/10.1145/3677052.3698621).


## Authors

Main author:

- [Simone Brusatin](https://github.com/Brusa99) <[simone.brusatin@gmail.com](mailto:simone.brusatin@gmail.com)>

Co-authors:

- [Aldo Glielmo](https://github.com/AldoGl) <[aldo.glielmo@bancaditalia.it](mailto:aldo.glielmo@bancaditalia.it)>

- [Andrea Coletta](https://github.com/Andrea94c) <[andrea.coletta@bancaditalia.it](mailto:andrea.coletta@bancaditalia.it)>

- [Tommaso Padoan](https://github.com/tpadoan) <[tommaso.padoan@units.it](mailto:tommaso.padoan@units.it)>


