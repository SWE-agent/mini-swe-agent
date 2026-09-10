{
  description = "mini-SWE-agent: the 100 line AI agent that solves GitHub issues";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    let
      inherit (nixpkgs) lib;

      forAllSystems = lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      pythonSets = forAllSystems (
        system:
        (nixpkgs.legacyPackages.${system}.callPackage pyproject-nix.build.packages {
          python = nixpkgs.legacyPackages.${system}.python3;
        }).overrideScope
          (lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            overlay
          ])
      );
    in
    {
      packages = forAllSystems (
        system:
        let
          pythonSet = pythonSets.${system};
          inherit (nixpkgs.legacyPackages.${system}.callPackages pyproject-nix.build.util { }) mkApplication;
        in
        {
          mini-swe-agent = mkApplication {
            venv = pythonSet.mkVirtualEnv "mini-swe-agent-env" workspace.deps.default;
            package = pythonSet.mini-swe-agent;
          };
        }
      );

      apps = forAllSystems (system: {
        mini-swe-agent = {
          type = "app";
          program = "${self.packages.${system}.mini-swe-agent}/bin/mini-swe-agent";
          meta.description = "The mini-SWE-agent CLI";
        };
        mini-extra = {
          type = "app";
          program = "${self.packages.${system}.mini-swe-agent}/bin/mini-extra";
          meta.description = "Extra utilities for mini-SWE-agent (benchmarks, config, inspector)";
        };
      });

      checks = forAllSystems (system: {
        mini-swe-agent = self.packages.${system}.mini-swe-agent;
      });
    };
}
