## Physics-informed neural networks for highly compressible flows

by Thomas Wagenaar

### Abstract
While physics-informed neural networks have been shown to accurately solve a wide range of fluid dynamics problems, their effectivity on highly compressible flows is so far limited. In particular, they struggle with transonic and supersonic problems that involve discontinuities such as shocks. While there have been multiple efforts to alleviate the nonphysical phenomena that arise on such problems, current work does not identify and address the underlying failure modes sufficiently. This thesis shows that physics-informed neural networks struggle with highly compressible problems for two independent reasons. Firstly, the differential Euler equations conserve entropy along streamlines, so that physics-informed neural networks try to find an isentropic solution to a non-isentropic problem. Secondly, conventional slip boundary conditions form strong local minima that result in fictive objects that simplify the flow. In response to these failure modes, two new adaptations are introduced, namely a local viscosity method and a streamline output representation. The local viscosity method includes viscosity as an additional network output and adds a viscous loss term to the loss function, resulting in localized viscosity that facilitates the entropy change at shocks. Furthermore, the streamline output representation provides a more natural formulation of the slip boundary conditions, which prevents zero-velocity regions while promoting shock attachment. To the author's best knowledge, this thesis provides the first inviscid steady solutions of curved and detached shocks by physics-informed neural networks.

![image](https://i.imgur.com/fbX3B4d.png)

### Report
Available at https://repository.tudelft.nl/islandora/object/uuid:6fd86786-153e-4c98-b4e2-8fa36f90eb2a. 

[<img src="https://i.imgur.com/Kybkxak.png" width="200">](https://repository.tudelft.nl/islandora/object/uuid:6fd86786-153e-4c98-b4e2-8fa36f90eb2a)

#### Errata

- Page 13, 'casual' should be 'causal'
- Page 42, Eq. (4.11) should be `-u sin(θ) + v cos(θ) = 0`
- Page 45, Eq. (5.1) should be `(-u sin(θ) + v cos(θ)) / (||u|| + ϵ) = 0`


## Source code
The folders `oblique`, `curved` and `detached` correspond to the problems treated in Chapter 6 respectively. Each folder contains a selection of the models used to generate the figures for each problem, if you would like to request the source code for a missing model, please request it by [creating an issue](https://github.com/wagenaartje/pinn4hcf/issues). Furthermore, each folder includes a `reference` folder that contains the code to generate the reference solution. 

### How to run
Simply run the `.py` files from the main folder of this repository. It will generate a pickled `.p` file that contains the history of the metrics and the network parameters at regular intervals. The file `utils.py` contains functions that can be used to load the pickled files. To reproduce the figures in the report, have a look at the `example_figure.py` file which shows how for example the density can be plotted for the detached shock wave.

### Requirements
```
torch
numpy
skopt
torch-cgd
```

## How to cite

```
@mastersthesis{Wagenaar2023,
  author  = {Wagenaar, Thomas},
  title   = {Physics-informed neural networks for highly compressible flows: assessing and enhancing shock-capturing capabilities},
  school  = {Delft University of Technology},
  year    = 2023,
  month   = {September}
}
```

