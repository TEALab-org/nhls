rm -rf gen_2d &&
mkdir gen_2d &&
python scripts/gen_2d_stencil.py &&
cargo run --release --example gen_2d &&
ffmpeg -framerate 30 -pattern_type glob -i 'gen_2d/frame*.png' -c:v libx264 -pix_fmt yuv420p gen_2d/out.mp4 &&
open gen_2d/out.mp4
