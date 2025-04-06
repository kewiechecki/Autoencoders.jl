{
  description = "Flake for Autoencoders.jl";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, "flake-utils": utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        # A derivation for your package.
        packages.autoencoders = pkgs.stdenv.mkDerivation {
          name = "Autoencoders.jl";
          src = ./.;
          # If your package is purely interpreted, no build phase is needed.
          # You can extend this if you have precompilation or other build steps.
        };

        # A development shell that provides Julia with your package instantiated.
        devShell = pkgs.mkShell {
          name = "autoencoders-dev-shell";
          buildInputs = [ pkgs.julia ];
          shellHook = ''
            echo "Entering Autoencoders.jl development shell..."
            # Activate the project and instantiate dependencies.
            julia --project=. -e 'using Pkg; Pkg.instantiate()'
          '';
        };
      }
    );
}

