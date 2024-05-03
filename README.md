# PYCATS

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
