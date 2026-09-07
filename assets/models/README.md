# Model Assets

The loader is generic and accepts ASCII PLY and OBJ model files. The Stanford
Bunny is the first demonstration asset, not a renderer-specific model type.

The `bunny/` directory comes from the Stanford Graphics Laboratory 3D Scan
Repository:

`https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz`

The interactive viewer uses `bunny/reconstruction/bun_zipper_res2.ply`, a
16,301-triangle decimated reconstruction. It keeps the full mesh silhouette
while remaining practical for the CPU renderer. Higher-resolution
reconstructions are retained in the same directory for offline renders.

The Stanford archive contains the original scan/reconstruction data and its
source README. Review the upstream repository terms before redistributing the
asset outside this project.
