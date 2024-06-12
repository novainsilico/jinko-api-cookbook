{
  description = "Jinko Cookbook Documentation";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  outputs =
    { self
    , flake-utils
    , nixpkgs
    , ...
    } @ inputs:
    flake-utils.lib.eachSystem
      [
        flake-utils.lib.system.x86_64-linux
      ]
      (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          shellBuildInputs = [
            # Poetry is the default package manager for the cookbook project
            pkgs.poetry

            # Add interactive bash to support `poetry shell`
            pkgs.bashInteractive
            pkgs.jq
          ];
        in
        {
          # Default shell with only poetry installed
          devShells = {
            default = pkgs.mkShell {
              name = "default";
              buildInputs = shellBuildInputs;
              shellHook = ''
                export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc
                ]}
                git config filter.cleanup-notebook.clean 'scripts/cleanup-notebook.sh'
                git config filter.cleanup-notebook.smudge 'cat'
              '';
            };

            # Install and load a poetry shell
            poetry = pkgs.mkShell {
              buildInputs = shellBuildInputs;
              shellHook = ''
                export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc
                ]}
                poetry install
                poetry shell
                git config filter.cleanup-notebook.clean 'scripts/cleanup-notebook.sh'
                git config filter.cleanup-notebook.smudge 'cat'
              '';
            };

            # Run jupyter lab in a poetry shell
            lab = pkgs.mkShell {
              buildInputs = shellBuildInputs;
              shellHook = ''
                export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc
                ]}
                poetry install
                poetry run jupyter-lab
                git config filter.cleanup-notebook.clean 'scripts/cleanup-notebook.sh'
                git config filter.cleanup-notebook.smudge 'cat'
              '';
            };

          };
        }
      );
}
