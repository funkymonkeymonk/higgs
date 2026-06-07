{ config, lib, pkgs, ... }:
with lib;
let
  cfg = config.services.higgs;
  tomlFormat = pkgs.formats.toml { };

  clean = v:
    if isAttrs v && !isList v then
      filterAttrs (_: val: val != null && val != "") (mapAttrs (_: clean) v)
    else
      v;

  serverConfig = clean {
    host = cfg.server.host;
    port = cfg.server.port;
    max_tokens = cfg.server.maxTokens;
    timeout = cfg.server.timeout;
    max_body_size = cfg.server.maxBodySize;
    rate_limit = cfg.server.rateLimit;
  };

  localConfig = clean {
    mlx_profile = cfg.local.mlxProfile;
    raise_wired_limit = cfg.local.raiseWiredLimit;
  };

  modelsConfig = map (m:
    clean {
      path = "${m.package}";
      name = m.name;
      mlx_profile = m.mlxProfile;
      batch = m.batch;
      kv_cache = m.kvCache;
    }) cfg.models;

  providerNames = builtins.attrNames cfg.providers;
  providersConfig = mapAttrs (_: p: clean {
    url = p.url;
    format = p.format;
    api_key = p.apiKey;
  }) cfg.providers;

  routesConfig = map (r:
    clean {
      pattern = r.pattern;
      provider = r.provider;
      model = r.model;
    }) cfg.routes;

  tomlConfig =
    { server = serverConfig; }
    // optionalAttrs (localConfig != { }) { local = localConfig; }
    // optionalAttrs (modelsConfig != [ ]) { models = modelsConfig; }
    // optionalAttrs (providersConfig != { }) { provider = providersConfig; }
    // optionalAttrs (routesConfig != [ ]) { routes = routesConfig; }
    // {
      default = {
        provider = cfg.defaultProvider;
      };
    };

in
{
  options.services.higgs = {
    enable = mkEnableOption "Higgs LLM inference server";

    package = mkOption {
      type = types.package;
      default = pkgs.higgs;
      defaultText = literalExpression "pkgs.higgs";
      description = "Higgs package to use";
    };

    server = {
      host = mkOption {
        type = types.str;
        default = "0.0.0.0";
        description = "Bind address";
      };
      port = mkOption {
        type = types.port;
        default = 8000;
        description = "Bind port";
      };
      maxTokens = mkOption {
        type = types.nullOr types.ints.unsigned;
        default = null;
        description = "Maximum generation tokens";
      };
      timeout = mkOption {
        type = types.nullOr types.number;
        default = null;
        description = "Request timeout in seconds";
      };
      maxBodySize = mkOption {
        type = types.nullOr types.ints.unsigned;
        default = null;
        description = "Maximum request body size in bytes";
      };
      rateLimit = mkOption {
        type = types.nullOr types.ints.unsigned;
        default = null;
        description = "Requests per minute per client";
      };
    };

    local = {
      mlxProfile = mkOption {
        type = types.enum [ "auto" "latency" "balanced" "throughput" ];
        default = "auto";
        description = "Default MLX tuning profile for local models";
      };
      raiseWiredLimit = mkOption {
        type = types.bool;
        default = false;
        description = "Allow MLX to raise the process wired-memory limit";
      };
    };

    models = mkOption {
      type = types.listOf (types.submodule {
        options = {
          name = mkOption {
            type = types.str;
            description = "Short name for the model";
          };
          package = mkOption {
            type = types.package;
            description = "Nix package providing the model";
          };
          mlxProfile = mkOption {
            type = types.nullOr (types.enum [ "auto" "latency" "balanced" "throughput" ]);
            default = null;
            description = "Per-model MLX profile override";
          };
          batch = mkOption {
            type = types.nullOr types.bool;
            default = null;
            description = "Enable continuous batching";
          };
          kvCache = mkOption {
            type = types.nullOr (types.enum [ "off" "turboquant" ]);
            default = null;
            description = "KV cache mode";
          };
        };
      });
      default = [ ];
      description = "Local MLX models to serve";
    };

    providers = mkOption {
      type = types.attrsOf (types.submodule {
        options = {
          url = mkOption {
            type = types.str;
            description = "Provider API base URL";
          };
          format = mkOption {
            type = types.str;
            default = "openai";
            description = "API format";
          };
          apiKey = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "API key";
          };
        };
      });
      default = { };
      description = "Remote LLM providers to proxy";
    };

    routes = mkOption {
      type = types.listOf (types.submodule {
        options = {
          pattern = mkOption {
            type = types.str;
            description = "Model name pattern to match";
          };
          provider = mkOption {
            type = types.str;
            description = "Provider to route matching requests to";
          };
          model = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Override model name sent to provider";
          };
        };
      });
      default = [ ];
      description = "Routing rules for model requests";
    };

    defaultProvider = mkOption {
      type = types.str;
      default = "higgs";
      description = "Default provider for model requests";
    };
  };

  config = mkIf cfg.enable {
    assertions = [{
      assertion = (cfg.models != [ ]) || (cfg.providers != { });
      message = "Higgs requires at least one local model or remote provider";
    }];

    environment.systemPackages = [ cfg.package ];

    # Generate TOML config
    environment.etc."higgs/config.toml" = mkIf (cfg.models != [ ] || cfg.providers != { }) {
      source = tomlFormat.generate "config.toml" tomlConfig;
    };
  };
}
