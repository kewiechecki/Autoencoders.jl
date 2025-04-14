using Pkg
Pkg.instantiate()

Pkg.add("cuDNN")
Pkg.add("StructArrays")

Pkg.precompile()
using Autoencoders, cuDNN

