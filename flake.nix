{
  description = "Jinko Cookbook Documentation";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.jinko-seeder.url = "git+ssh://git@git.novadiscovery.net/jinko/dorayaki/jinko-seeder";
  inputs.dev-seeding.url = "git+ssh://git@git.novadiscovery.net/jinko/dorayaki/dango.git?dir=e2e/src";
  outputs =
    { self
    , flake-utils
    , nixpkgs
    , jinko-seeder
    , dev-seeding
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
            pkgs.nodejs
            pkgs.python312
          ];
          shellInit = ''
            source .envrc 2> /dev/null || true
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
            ]}
            export POETRY_CACHE_DIR="./.cache/pypoetry"
            git config filter.cleanup-notebook.clean 'scripts/cleanup-notebook.sh'
            git config filter.cleanup-notebook.smudge 'cat'
            source .envrc
          '';
        in
        {
          # Default shell with only poetry installed
          devShells = {
            default = pkgs.mkShell {
              name = "default";
              buildInputs = shellBuildInputs;
              shellHook = ''
                ${shellInit}
              '';
            };

            # Install and load a poetry shell
            poetry = pkgs.mkShell {
              buildInputs = shellBuildInputs;
              shellHook = ''
                ${shellInit}
                poetry env use ${pkgs.python312}/bin/python
                poetry install
                source $(poetry env info --path)/bin/activate
              '';
            };

            # Run jupyter lab in a poetry shell
            lab = pkgs.mkShell {
              buildInputs = shellBuildInputs;
              shellHook = ''
                ${shellInit}
                poetry env use ${pkgs.python312}/bin/python
                poetry install
                poetry run jupyter-lab
              '';
            };

            # Run e2e tests
            e2e = pkgs.mkShell {
              buildInputs = shellBuildInputs ++ [ 
                dev-seeding.packages.${system}.dev-seeding
                jinko-seeder.packages.${system}.jinko-seeder
              ];
              shellHook = ''
                ${shellInit}
                poetry env use ${pkgs.python312}/bin/python
                poetry install
                export jinko_seeder=${jinko-seeder}
                source ${jinko-seeder}/jinko-seeder.bash
                export seeding=${dev-seeding}/scripts
                # this replaces calling poetry shell
                VENV_PATH=$(poetry env info --path)
                source $VENV_PATH/bin/activate
                cd e2e
              '';
            };

          };
        }
      );
}
