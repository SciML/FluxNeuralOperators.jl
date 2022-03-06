# Super Resolution

The time dependent Navier-Stokes equation is learned by the `MarkovNeuralOperator` with only one time step information.
The result of this example can be found [here](https://neuraloperators.sciml.ai/dev/assets/notebook/super_resolution_mno.jl.html).

Apart from just training a MNO, here, we train the model with lower resolution (96x64) and inference result with higher resolution (192x128).

| **Ground Truth** | **Inferenced** |
|:----------------:|:--------------:|
| ![](gallery/ans.gif) | ![](gallery/inferenced.gif) |

Change directory to `example/SuperResolution` and use following commend to train model:

```julia
$ julia --proj

julia> using SuperResolution; SuperResolution.train()
```
