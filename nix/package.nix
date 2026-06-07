{
  lib,
  stdenv,
  rustPlatform,
  darwin,
  pkg-config,
  ...
}:
let
  version = "1.3.0";
in
rustPlatform.buildRustPackage {
  pname = "higgs";
  inherit version;

  src = lib.cleanSourceWith {
    src = ../.;
    filter = path: type:
      (lib.hasSuffix ".rs" path)
      || (lib.hasSuffix ".toml" path)
      || (lib.hasSuffix ".lock" path)
      || (lib.hasSuffix ".json" path)
      || (lib.hasSuffix ".md" path)
      || (lib.hasSuffix ".html" path)
      || (lib.hasSuffix ".css" path)
      || builtins.elem (baseNameOf path) [
        "Cargo.toml"
        "Cargo.lock"
        "rustfmt.toml"
        "build.rs"
      ];
  };

  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [ pkg-config ];

  buildInputs =
    lib.optionals stdenv.isDarwin
      (with darwin.apple_sdk.frameworks; [
        Metal
        MetalKit
        CoreGraphics
        CoreVideo
        CoreImage
        Accelerate
        Foundation
      ]);

  env = lib.optionalAttrs stdenv.isDarwin {
    METAL_DEVICE_WRAPPER_TYPE = "1";
    CORESERVICES_FRAMEWORK_SEARCH = "YES";
  };

  doCheck = false;

  meta = with lib; {
    description = "Local LLM inference server for Apple Silicon using MLX";
    homepage = "https://github.com/panbanda/higgs";
    license = licenses.mit;
    platforms = [ "aarch64-darwin" ];
    maintainers = with maintainers; [ ];
    mainProgram = "higgs";
  };
}
