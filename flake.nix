{
  description = "Flake for Autoencoders.jl";
  nixConfig = {
    bash-prompt = "\\[Autoencoders$(__git_ps1 \" (%s)\")\\]$ ";
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
          config = {
            allowUnfree = true;
            cudaSupport = system == "x86_64-linux";
          };
        };

        juliaPkgs = pkgs.juliaPackages;

        # Define the list of packages for the shell environment
        shellPkgsNested = with pkgs; [
          julia
          git
          stdenv.cc       # Standard compiler environment (needed for wrappers/headers)
          gfortran      # The gfortran *wrapper* executable
          stdenv.cc.cc.lib # <--- The actual GCC runtime libraries (incl. libquadmath)
          (lib.optional stdenv.isLinux cudaPackages.cudatoolkit)
          (lib.optional stdenv.isLinux cudaPackages.cudnn)
        ];

        # Flatten the list
        shellPkgs = pkgs.lib.flatten shellPkgsNested;

        # Build Autoencoders.jl
        autoencodersBuilt = juliaPkgs.buildJuliaPackage {
          pname = "Autoencoders";
          version = "0.1.1"; # TODO: FIX THIS
          src = ./.;
          # Propagate the actual libs maybe? Or just gfortran wrapper?
          # Let's keep gfortran for now, assuming julia build might need wrapper.
          propagatedBuildInputs = [ pkgs.gfortran ];
        };

      in {
        packages.autoencoders = autoencodersBuilt;
        packages.default = self.packages.${system}.autoencoders;

        devShell = pkgs.mkShell {
          name = "autoencoders-dev-shell";
          # Use the list including stdenv.cc.cc.lib
          buildInputs = shellPkgs;

          shellHook = ''
            source ${pkgs.git}/share/bash-completion/completions/git-prompt.sh
            export JULIA_PROJECT="@."

            # Trust makeLibraryPath again now that stdenv.cc.cc.lib is included
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath shellPkgs}";

            echo "Nix dev shell for Autoencoders.jl activated."
            echo "Julia environment uses Project.toml (JULIA_PROJECT=@.)."
            # Debug check again - THIS SHOULD FINALLY FIND IT
            echo "--- Checking LD_LIBRARY_PATH ($LD_LIBRARY_PATH) for libquadmath.so.0 ---"
            ( IFS=: ; for p in $LD_LIBRARY_PATH; do if [ -f "$p/libquadmath.so.0" ]; then echo "  FOUND in $p"; fi; done )
            echo "----------------------------------------------------"
          '';
        };
      }
    );
}
