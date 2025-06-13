# Non-homongeneous Linear Stencils (NHLS)

![Rust workflow](https://github.com/SallySoul/nhls/actions/workflows/rust.yml/badge.svg?branch=main)

This repo is for exploring ways to solve NHLS problems.

## For Developers

Control thread number with 
```
export RAYON_NUM_THREADS=n
```

### Installing Rust

If this is your first time using rust,
you will need to install the rust toolchain.
This is commonly done with a utility called `rustup`,
which allows you download and update multiple versions
of the rust toolchains.

For the latest details on obtaining and using `rustup`,
please refer to the official [Install Rust](https://www.rust-lang.org/tools/install).

Some notable utilities managed by `rustup` are:
* `rustc` - The actual compiler. Its likely you will never interact with this program directly.
* `cargo` - The standard build system and package manager for rust projects. 
  Most actions you will want to run will be of the form `cargo <action>`.
* `rustanalyzer` - This component implements an 
  [LSP](https://microsoft.github.io/language-server-protocol/)
  server to provide and IDE like experience in your editor of choice.

### Useful Resources

The Rust ecosystem has some great documentation. 
The [official website](https://www.rust-lang.org) is a good place to start.
There are several resources there for learning rust, including a textbook and a by-example tutorial.

The standard library documentation can be found [here](https://doc.rust-lang.org/std/index.html).

Rust packages are called crates, 
and [crates.io](https://crates.io) is the official crate registry.
Note that each crate's page includes links to both its github,
and documentation hosted on [docs.rs](https://docs.rs).

### Using Cargo

This project is setup as a rust library.
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

Cargo also includes a source formatter.
```text
cargo fmt
```

Lastly, library projects can include example executables, 
included in the examples directory.
For example, to run the file `examples/example_1.rs`
```text
cargo run --example example_1           # Debug 
cargo run --release --example example_1 # Release, recommended
```

### Profiling

NHLS and the [fftw3-bindings](https://github.com/SallySoul/fftw3-rs) are instrumented with macros from the [profiling](https://github.com/aclysma/profiling) crate.
When built with the `profiling-with-puffin` feature,
examples become clients for the [Puffin](https://github.com/EmbarkStudios/puffin) profiler.
In the following example we profile `heat_2d_ap_fft` using `puffin_viewer`.

```bash
# Start the puffin viewer.
# You can run it from source with:
cargo run --bin puffin_viewer --release

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
