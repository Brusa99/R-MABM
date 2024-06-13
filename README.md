# PyCATS

A python implementation of the "CATS" model (Emergent dynamics of a macroeconomic agent based model with capital and credit, Assenza et al. (2017)).


## Installation

After having installed the appropriate dependencies, run in python:

```python
import juliapkg as pkg

pkg.add("Agents", uuid="46ada45e-f475-11e8-01d0-f70cc89e6671", version=5.17)
pkg.add("Cats", "91b5283e-3ff1-4617-8122-130c88cb8aba", url="git@github.com:bancaditalia/Cats.jl.git")

pkg.resolve()

# if everything went well, you should see the packages in the environment by running
pkg.status()
```

### Authors

Main author:

- [Simone Brusatin](https://github.com/Brusa99) <[simone.brusatin@studenti.units.it](mailto:simone.brusatin@studenti.units.it)>

Supervisors:

- [Aldo Glielmo](https://github.com/AldoGl) <[aldo.glielmo@bancaditalia.it](mailto:aldo.glielmo@bancaditalia.it)>

- [Andrea Coletta](https://github.com/Andrea94c) <[andrea.coletta@bancaditalia.it](mailto:andrea.coletta@bancaditalia.it)>

- [Tommaso Padoan](https://github.com/tpadoan) <[tommaso.padoan@units.it](mailto:tommaso.padoan@units.it)>


## Disclaimer

This package is an outcome of a research project.
A preprint of the article is available [here](https://arxiv.org/abs/2405.02161).
