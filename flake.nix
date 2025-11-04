{
  description = "nix develop flake for beatstrip audio transform";

  inputs ={
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
  flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          pkg-config
        ];

        buildInputs = with pkgs; [
          alsa-lib
        ];

        LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [
          libGL
          libxkbcommon
          wayland
        ];
      };
    }
  );
}
