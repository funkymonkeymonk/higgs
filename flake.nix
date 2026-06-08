{
  description = "Higgs - local LLM inference server for Apple Silicon (MLX)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" ] (system:
      let
        # Overlay to provide apple_sdk_11_0 stub (removed in nixpkgs 26.11+)
        # darwin stdenv __impureHostDeps still references it
        fixAppleSdk = final: prev: let
          mkStub = name: prev.stdenv.mkDerivation {
            name = "${name}-stub";
            phases = [ "installPhase" ];
            installPhase = ''
              mkdir -p $out/Library/Frameworks/${name}.framework/Versions/Current
            '';
          };
        in {
          darwin = prev.darwin // {
            apple_sdk_11_0 = {
              Foundation = mkStub "Foundation";
              IOKit = mkStub "IOKit";
              Libsystem = mkStub "Libsystem";
            };
          };
        };

        pkgs = import nixpkgs {
          inherit system;
          config.allowUnsupportedSystem = true;
          overlays = [ fixAppleSdk ];
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
      darwinModules.higgs = import ./nix/darwin-module.nix;
    };
}
