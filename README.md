# Non-homongeneous Linear Stencils (NHLS)

![Rust workflow](https://github.com/SallySoul/nhls/actions/workflows/rust.yml/badge.svg?branch=main)

This repo is for exploring ways to solve NHLS problems. We are also developing a corresponding python module [`nhls_py`](https://github.com/TEALab-org/nhls_py) for utilizing the solvers from this project.

## Example

A number of example executables are included that demonstrate different linear stencil problems.
They provide a CLI interface, try using `--help` for more information.
Consider this example where we generate a short animation of a time-varying stencil computation.

```
cargo run \
    --example tv_rotating_2d \
    --release \
    -- \
    --wisdom-file target/tv_rotating_2d/wisdom \
    --write-images target/tv_rotating_2d \
    --plan-type measure \
    --domain-size 1000 \
    --steps-per-image 1000 \
    --images 31 \
    --ic-type impulse \
    --ic-dial 10 \
    --task-min 4 \
    --task-mult 2 \
    --threads 8
    
cd target/tv_rotating_2d
ffmpeg \
	-framerate 30 \
	-pattern_type glob -i 'frame_*.png' \
	-c:v libx264 \
	-pix_fmt yuv420p \
	out.mp4
open out.mp4
```

## For Developers

NHLS requires using the nightly Rust toolchain.
Please refer to the official [Install Rust](https://www.rust-lang.org/tools/install) guide for obtaining `rustup`, the rust toolchain manager.
All changes should be made through pull requests, which must pass CI.

### Cargo Commands

To build the library use
```text
cargo build           # Debug build
cargo build --release # Release Build
```

Note that the first time you build a project,
all the dependencies need to be downloaded and built as well.
This is mostly a one time expense, as these will all be saved locally.

Tests can be run with 
```text
cargo test
```

Documentation for our library and our dependencies can be generated with
```text
cargo doc
cargo doc --open # Opens the index.html page
```

Cargo also includes a source formatter, which is required to pass CI
```text
cargo fmt
```

Lastly, library projects can include example executables, 
included in the examples directory.
For example, to run the file `examples/example_1.rs`
```text
cargo run --example example_1  # Debug 

cargo run --release \          # Release, recommended
    --example example_1 \      # Runs examples/example_1.rs
    -- \                       # Args following -- are passed to example
    --example-arg1
    --example-arg2
```

### Profiling

NHLS and the [fftw3-bindings](https://github.com/SallySoul/fftw3-rs) are instrumented with macros from the [profiling](https://github.com/aclysma/profiling) crate.
When built with the `profiling-with-puffin` feature,
examples become clients for the [Puffin](https://github.com/EmbarkStudios/puffin) profiler.
In the following example we profile `heat_2d_ap_fft` using `puffin_viewer`.

```bash
# Start the puffin viewer.
# You can run it from source with:
# Run in the background, or on another terminal
puffin_viewer&

# Here we run an expensive operation
cargo run \
    --example heat_2d_ap_fft \
    --release \
    --features profile-with-puffin \  # The image_*_example start a puffin client
    -- \
    --domain-size 4000 \
    --steps-per-image 4000 \
    --images 2 \                      # We just want one solver application
    --threads 8

# Wait for the example to finish running.
# Sometimes the profiling takes a while to buffer before it starts streaming
```

Note that when building with the `profile-with-puffin` feature, executables will have an additional `--puffin-url` flag that can be used to set a custom url if the default one isn't preferred (`127.0.0.1:8585`).
