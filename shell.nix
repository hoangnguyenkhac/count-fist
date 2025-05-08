{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.pip
    python3Packages.opencv4  # OpenCV dependency
    libGL                    # libGL dependency for OpenCV
  ];

  shellHook = ''
    export PYTHONPATH=$PYTHONPATH:${toString pkgs.python3.sitePackages}
  '';
}
