{
  description = "Higgs - local LLM inference server for Apple Silicon (MLX)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnsupportedSystem = true;
        };
      in
      {
        packages = {
          higgs = pkgs.callPackage ./nix/package.nix { };
          default = self.packages.${system}.higgs;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.higgs ];
          nativeBuildInputs = with pkgs; [
            rustc
            cargo
            rust-analyzer
            rustfmt
            clippy
          ];
        };
      }
    ) // {
      nixosModules.higgs = import ./nix/module.nix;
      darwinModules.higgs = import ./nix/module.nix;
    };
}
