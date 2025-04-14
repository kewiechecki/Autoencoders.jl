{
  description = "Flake for Autoencoders.jl";
  nixConfig = {
    bash-prompt = "\[Autoencoders$(__git_ps1 \" (%s)\")\]$ ";
  };

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = system == "x86_64-linux";
		};

        # Get library paths from the stdenv compiler and from gfortran.
        gccPath = toString pkgs.stdenv.cc.cc.lib;
        gfortranPath = toString pkgs.gfortran;

        # Define the multi-line Julia script.
        # NOTE: The closing delimiter (two single quotes) MUST be flush with the left margin.
        juliaScript = ''
using Pkg
Pkg.instantiate()

Pkg.add("cuDNN")
Pkg.add("StructArrays")

Pkg.precompile()
using Autoencoders, cuDNN
'';

		
      in {
        # A derivation for your package.
        packages.autoencoders = pkgs.stdenv.mkDerivation {
          name = "Autoencoders.jl";
          src = ./.;
          # If your package is purely interpreted, no build phase is needed.
          # You can extend this if you have precompilation or other build steps.
        };

        # A development shell that provides Julia with your package instantiated.
        devShell = with pkgs; mkShell {
          name = "autoencoders-dev-shell";
          buildInputs = [ 
		    julia 
			git
			stdenv.cc
			gfortran
		  ];
          shellHook = ''
source ${git}/share/bash-completion/completions/git-prompt.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${gfortranPath}/lib:${gccPath}/lib:${gccPath}/lib64
echo $LD_LIBRARY_PATH

cat > julia_deps.jl <<'EOF'
${juliaScript}
EOF

# Activate the project and instantiate dependencies.
julia --project=. julia_deps.jl
          '';
        };
      }
    );
}

