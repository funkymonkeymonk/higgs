{
  lib,
  stdenvNoCC,
  runCommandLocal,
  rustPlatform,
  pkg-config,
  cmake,
  git,
  zsh,
  fetchFromGitHub,
  fetchurl,
  fmt,
  ...
}:
let
  version = "1.3.0";

  json-tarball = fetchurl {
    url = "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz";
    sha256 = "0zbmh5gdbj91y5pfsq170c1y8yhhgcjmfirg31x8xmhydg55minn";
  };

  mlx-src = fetchFromGitHub {
    owner = "ml-explore";
    repo = "mlx";
    rev = "v0.30.6";
    hash = "sha256-avD5EGhwgmPdXLAyQSqTO6AXk/W3ziH+f6AetjK3Sdo=";
  };

  # Patch MLX's CMakeLists.txt: use /bin/zsh (absolute path) instead of bare zsh
  # since nix sandbox may not have /etc/shells or zsh in PATH for interactive use
  mlx-src-patched = runCommandLocal "mlx-source-patched" {} ''
    cp -r ${mlx-src} $out
    chmod -R +w $out
    # Replace "COMMAND zsh" with "COMMAND /bin/zsh" (both xcrun version check
    # and metal version check use this pattern)
    sed -i 's/COMMAND zsh/COMMAND \/bin\/zsh/g' $out/CMakeLists.txt
  '';
in
rustPlatform.buildRustPackage {
  pname = "higgs";
  inherit version;

  src = ../.;

  cargoLock = {
    lockFile = ../Cargo.lock;
    outputHashes = {
      "mlx-internal-macros-0.25.3" = "sha256-JAHA0JXQ/TnF0Yp86rztIOicgGoF6tvTJRaVQbRysNc=";
    };
  };

  nativeBuildInputs = [ pkg-config cmake git zsh ];

  buildInputs = [ fmt ];

  preConfigure = ''
    # Hardcode MLX_BUILD_METAL=OFF in mlx-sys build.rs (no Metal compiler without full Xcode)
    MLX_SYS_BUILD=$(find "$NIX_BUILD_TOP" -path "*/mlx-sys-*/build.rs" -print -quit)
    if [ -n "$MLX_SYS_BUILD" ]; then
      echo "Disabling metal build in mlx-sys build.rs at $MLX_SYS_BUILD"
      sed -i 's/config.define("MLX_BUILD_METAL", "ON")/config.define("MLX_BUILD_METAL", "OFF")/' "$MLX_SYS_BUILD"
    fi

    # Copy patched MLX source to a writable location (nix store is read-only)
    MLX_WORKDIR="$NIX_BUILD_TOP/mlx-source"
    cp -r ${mlx-src-patched} "$MLX_WORKDIR"
    chmod -R +w "$MLX_WORKDIR"

    # Patch MLX's CMakeLists.txt to use local copies of fetched dependencies
    # (network not available in nix sandbox)
    JSON_DIR="$NIX_BUILD_TOP/json-source"
    mkdir -p "$JSON_DIR"
    tar xf ${json-tarball} -C "$JSON_DIR" --strip-components=1

    # json: use SOURCE_DIR instead of URL download
    substituteInPlace "$MLX_WORKDIR/CMakeLists.txt" \
      --replace-fail 'URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz' \
                     "SOURCE_DIR \"$JSON_DIR\""

    # fmt: use system-installed fmt library instead of FetchContent
    substituteInPlace "$MLX_WORKDIR/CMakeLists.txt" \
      --replace-fail 'option(USE_SYSTEM_FMT "Use system'"'"'s provided fmt library" OFF)' \
                     'set(USE_SYSTEM_FMT ON)'

    # Disable GGUF (requires additional FetchContent for gguflib, not needed by Higgs)
    substituteInPlace "$MLX_WORKDIR/CMakeLists.txt" \
      --replace-fail 'option(MLX_BUILD_GGUF "Include support for GGUF format" ON)' \
                     'set(MLX_BUILD_GGUF OFF)'

    # Patch mlx-c wrapper's CMakeLists.txt to use pre-fetched MLX source
    # instead of git cloning at build time
    MLX_C_CMAKE=$(find "$NIX_BUILD_TOP" -path "*/mlx-sys-*/src/mlx-c/CMakeLists.txt" -print -quit)
    if [ -n "$MLX_C_CMAKE" ]; then
      echo "Patching mlx-c CMakeLists.txt at $MLX_C_CMAKE"
      substituteInPlace "$MLX_C_CMAKE" \
        --replace-fail 'GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"' \
                       "SOURCE_DIR \"$MLX_WORKDIR\"" \
        --replace-fail 'GIT_TAG v0.30.6' \
                       ""
    fi
  '';

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
