{ pkgs, lib, config, inputs, ... }:

{
  env.RUST_TEST_THREADS = "1";

  languages.rust = {
    enable = true;
    channel = "stable";
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" "rust-src" ];
  };

  packages = with pkgs; [
    git
    pkg-config
    cmake
    zsh
    fmt
    watchexec
    just
  ];

  scripts = {
    build.exec = "cargo build";
    check.exec = "cargo clippy -p higgs";
    fmt-check.exec = "cargo fmt -p higgs -- --check";
    test-higgs.exec = "cargo test -p higgs -- --test-threads=1";
    test-all.exec = "cargo test -- --test-threads=1";
    dev.exec = "cargo run --bin higgs";
  };

  processes = {
    higgs-server.exec = "cargo run --bin higgs";
    higgs-server-dev = {
      exec = "watchexec -r -e rs,toml -- cargo run --bin higgs";
    };
  };

  git-hooks.hooks = {
    clippy.enable = true;
    rustfmt.enable = true;
  };

  enterShell = ''
    echo "Higgs development environment"
    echo ""
    echo "  build            — cargo build"
    echo "  check            — cargo clippy -p higgs"
    echo "  fmt-check        — cargo fmt -p higgs -- --check"
    echo "  test-higgs       — cargo test -p higgs"
    echo "  test-all         — cargo test (all crates)"
    echo "  dev              — cargo run --bin higgs"
    echo ""
    echo "Processes (devenv up):"
    echo "  higgs-server     — cargo run --bin higgs"
    echo "  higgs-server-dev — watchexec hot-reload (cargo run --bin higgs)"
  '';

  enterTest = ''
    cargo test -p higgs -- --test-threads=1
  '';
}
