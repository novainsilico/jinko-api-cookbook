{
  description = "Jinko API Documentation";
  nixConfig.extra-substituters = [
    "https://tweag-jupyter.cachix.org"
  ];
  nixConfig.extra-trusted-public-keys = [
    "tweag-jupyter.cachix.org-1:UtNH4Zs6hVUFpFBTLaA4ejYavPo5EFFqgd7G7FxGW9g="
  ];
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.jupyenv.url = "github:tweag/jupyenv";
  outputs =
    { self
    , flake-utils
    , nixpkgs
    , jupyenv
    , ...
    } @ inputs:
    flake-utils.lib.eachSystem
      [
        flake-utils.lib.system.x86_64-linux
      ]
      (
        system:
        let
          inherit (jupyenv.lib.${system}) mkJupyterlabNew;
          pkgs = import nixpkgs { inherit system; };
          jupyterlab = mkJupyterlabNew ({ ... }: {
            nixpkgs = inputs.nixpkgs;
            imports = [
              ({ pkgs, ... }: {
                kernel.python.jinko = {
                  enable = true;
                  extraPackages = ps: [
                    ps.numpy
                    ps.scipy
                    ps.matplotlib
                    ps.requests
                  ];
                };
              })
            ];
          });
        in
        {
          packages = {
            inherit jupyterlab;
            git-lfs = nixpkgs.legacyPackages.${system}.git-lfs; # Add git-lfs here
          };
          packages.default = jupyterlab;
          apps.default.program = "${jupyterlab}/bin/jupyter-lab";
          apps.default.type = "app";
          # kernel.python.aiml.enable = true;

          # Add a shell with python and pip:
          devShell = pkgs.mkShell {
            buildInputs = [
              # Offer python, pip and poetry 
              # for people who want to install additional packages
              # using there own way
              pkgs.python3
              pkgs.poetry
              pkgs.python3Packages.pip

              # Add jupyter notebook command line tool
              pkgs.jupyter

              # Add interactive bash to support `poetry shell`
              pkgs.bashInteractive

            ];
            shellHook = ''
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc
              ]}
            '';
          };
        }
      );
}
