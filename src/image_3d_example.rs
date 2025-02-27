use crate::build_info;
use crate::fft_solver::PlanType;
use crate::util::*;
use clap::Parser;
use std::path::PathBuf;

/// nhls 2D stencil executable
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Directory for output files, will be created.
    /// WARNING, if this Directory
    /// already exists, current contents will be removed.
    #[arg(short, long)]
    pub output_dir: Option<std::path::PathBuf>,

    /// Chunk size to use for parallelism.
    #[arg(short, long, default_value = "1000")]
    pub chunk_size: usize,

    /// How many images to output.
    #[arg(short, long, default_value = "100")]
    pub images: usize,

    /// How many steps to take per line.
    #[arg(short, long, default_value = "16")]
    pub steps_per_image: usize,

    /// Domain size, assume cube
    #[arg(short, long, default_value = "100")]
    pub domain_size: usize,

    /// Write out image, WARNING: we do not check image size, so be reasonable.
    #[arg(short, long, requires("output_dir"))]
    pub write_images: bool,

    /// The number of threads to use.
    #[arg(short, long, default_value = "8")]
    pub threads: usize,

    /// FFTW3 plan creation strategy.
    #[arg(short, long, default_value = "estimate")]
    pub plan_type: PlanType,

    /// File to load and save FFTW3 wisdom.
    #[arg(long)]
    pub wisdom_file: Option<PathBuf>,

    /// Fill with random values matching 2023 implementation
    #[arg(short, long)]
    pub rand_init: bool,

    /// Write out a dot file for the ap plan
    #[arg(long, requires("output_dir"))]
    pub write_dot: bool,

    /// Target ratio for fft solves
    #[arg(long, default_value = "0.5")]
    pub ratio: f64,

    /// Cutoff for fft solves
    #[arg(long, default_value = "40")]
    pub cutoff: i32,

    /// Generate solver only, do not solve
    #[arg(long)]
    pub gen_only: bool,

    /// Print build information and quit
    #[arg(long)]
    pub build_info: bool,
}

impl Args {
    pub fn cli_parse(name: &str) -> Self {
        let args = Args::parse();

        if args.build_info {
            build_info::print_report(name);
            std::process::exit(0);
        }

        if let Some(output_dir) = &args.output_dir {
            let output_dir = output_dir.to_str().unwrap();
            let _ = std::fs::remove_dir_all(output_dir);
            std::fs::create_dir(output_dir).unwrap();
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
        fftw::threading::init_threads_f64().unwrap();
        fftw::threading::plan_with_nthreads_f64(args.threads);

        if let Some(ref wisdom_path) = args.wisdom_file {
            if wisdom_path.exists() {
                fftw::wisdom::import_wisdom_file_f64(&wisdom_path).unwrap();
            }
        }

        args
    }

    pub fn grid_bounds(&self) -> AABB<3> {
        let inclusive = self.domain_size as i32 - 1;
        AABB::new(matrix![0, inclusive; 0, inclusive; 0, inclusive])
    }

    pub fn frame_name(&self, i: usize) -> PathBuf {
        let mut result = self.output_dir.as_ref().unwrap().clone();
        result.push(format!("frame_{:04}.vtu", i));
        result
    }

    pub fn dot_path(&self) -> PathBuf {
        let mut dot_path = self.output_dir.as_ref().unwrap().clone();
        dot_path.push("plan.dot");
        dot_path
    }

    pub fn save_wisdom(&self) {
        if let Some(ref wisdom_path) = self.wisdom_file {
            fftw::wisdom::export_wisdom_file_f64(&wisdom_path).unwrap();
        }
    }
}
