{ config, lib, pkgs, ... }:
with lib;
{
  imports = [ ./module.nix ];

  config = mkIf config.services.higgs.enable {
    launchd.user.agents.higgs = {
      command = "${config.services.higgs.package}/bin/higgs";
      serviceConfig = {
        ProgramArguments = [
          "${config.services.higgs.package}/bin/higgs"
          "serve"
          "--config"
          "/etc/higgs/config.toml"
        ];
        RunAtLoad = true;
        KeepAlive = true;
        Label = "org.higgs.server";
        WorkingDirectory = "/var/empty";
        ThrottleInterval = 5;
      };
    };
  };
}
