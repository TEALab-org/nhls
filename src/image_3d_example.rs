use crate::ap_solver::SolverParameters;
use crate::build_info;
use crate::domain::*;
use crate::fft_solver::PlanType;
use crate::image_example_util::*;
use crate::initial_conditions::*;
use crate::solver_interface::SolverInterface;
use crate::util::*;
use crate::vtk::*;
use clap::Parser;
use std::io::prelude::*;
use std::path::PathBuf;
use std::time::*;

#[cfg(feature = "profile-with-puffin")]
static PUFFIN_SERVER: std::sync::Mutex<Option<puffin_http::Server>> =
    std::sync::Mutex::new(None);

/// nhls 2D stencil executable
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
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
    #[arg(short, long)]
    pub write_images: Option<PathBuf>,

    /// The number of threads to use.
    #[arg(short, long, default_value = "8")]
    pub threads: usize,

    /// FFTW3 plan creation strategy.
    #[arg(short, long, default_value = "estimate")]
    pub plan_type: PlanType,

    /// File to load and save FFTW3 wisdom.
    #[arg(long)]
    pub wisdom_file: Option<PathBuf>,

    /// The type of initial condition to use.
    /// Some can be parameterized with --ic-dial <f64>.
    /// Use --ic-type rand --ic-dial 1024.0 to match 2023 implementation.
    #[arg(long, default_value = "zero")]
    pub ic_type: ClapICType,

    /// Parameter for initial_conditions
    #[arg(long)]
    pub ic_dial: Option<f64>,

    /// Fill with random values matching 2023 implementation
    #[arg(short, long)]
    pub rand_init: bool,

    /// Write out a dot file for the ap plan
    #[arg(long)]
    pub write_dot: Option<PathBuf>,

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

    /// Minimum number of tasks a plan node can use
    #[arg(long, default_value = "1")]
    pub task_min: usize,

    /// Assume total tasks available relative to threads
    #[arg(long, default_value = "1")]
    pub task_mult: f64,

    /// Write out the solver apply time in seconds to file
    #[arg(long)]
    pub timings_file: Option<PathBuf>,

    /// Puffin Viewer url
    #[cfg(feature = "profile-with-puffin")]
    #[arg(long, default_value = "127.0.0.1:8585")]
    pub puffin_url: String,
}

impl Args {
    pub fn solver_parameters(&self) -> SolverParameters<3> {
        let grid_bound = self.grid_bounds();
        SolverParameters {
            plan_type: self.plan_type,
            cutoff: self.cutoff,
            ratio: self.ratio,
            chunk_size: self.chunk_size,
            threads: self.threads,
            steps: self.steps_per_image,
            aabb: grid_bound,
            task_min: self.task_min,
            task_mult: self.task_mult,
        }
    }

    pub fn run_solver<SolverType: SolverInterface<3>>(
        &self,
        solver: &mut SolverType,
    ) {
        // Create domains
        let grid_bound = self.grid_bounds();
        let mut buffer_1 = OwnedDomain::new(grid_bound);
        let mut buffer_2 = OwnedDomain::new(grid_bound);
        let mut input_domain = buffer_1.as_slice_domain();
        let mut output_domain = buffer_2.as_slice_domain();

        self.run_solver_with_domains(
            &mut input_domain,
            &mut output_domain,
            solver,
        );
    }

    pub fn run_solver_with_domains<'a, SolverType: SolverInterface<3>>(
        &self,
        input_domain: &mut SliceDomain<'a, 3>,
        output_domain: &mut SliceDomain<'a, 3>,
        solver: &mut SolverType,
    ) {
        // Solver Diagnostics
        solver.print_report();
        if let Some(dot_path) = self.write_dot.as_ref() {
            solver.to_dot_file(&dot_path);
        }
        if self.gen_only {
            self.finish();
            std::process::exit(0);
        }

        // Setup initial conditions
        let ic_type = self.ic_type.to_ic_type(self.ic_dial);
        generate_ic_3d(input_domain, ic_type, self.chunk_size);
        if self.write_images.is_some() {
            write_vtk3d(input_domain, &self.frame_name(0));
        }

        // Setup timings file (maybe)
        let mut timings_writer = None;
        if let Some(timings_file) = self.timings_file.as_ref() {
            let writer = std::io::BufWriter::new(
                std::fs::File::create(timings_file).unwrap(),
            );
            timings_writer = Some(writer);
        }

        // Main solver loop
        let mut global_time = 0;
        for t in 1..self.images {
            // Run and time solver application
            let now = Instant::now();
            solver.apply(input_domain, output_domain, global_time);
            let elapsed_time = now.elapsed();
            let s_elapsed = elapsed_time.as_nanos() as f64 / 1000000000.0;
            println!("{s_elapsed}");
            if let Some(writer) = timings_writer.as_mut() {
                writeln!(writer, "{s_elapsed}").unwrap();
            }
            // Prepare for the next frame
            global_time += self.steps_per_image;
            std::mem::swap(input_domain, output_domain);

            // Write output (maybe)
            if self.write_images.is_some() {
                write_vtk3d(input_domain, &self.frame_name(t));
            }
        }

        self.finish();
    }

    pub fn cli_setup(name: &str) -> Self {
        let args = Args::parse();

        if args.build_info {
            build_info::print_report(name);
            std::process::exit(0);
        }

        #[cfg(feature = "profile-with-puffin")]
        {
            println!("Initializing profiling server:");
            let server = puffin_http::Server::new(&args.puffin_url).unwrap();
            println!(
                "Run this to view profiling data:  puffin_viewer {}",
                args.puffin_url
            );
            let mut server_lock = PUFFIN_SERVER.lock().unwrap();
            *server_lock = Some(server);
            profiling::puffin::set_scopes_on(true);
            profiling::finish_frame!();
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
        fftw::threading::init_threads_f64().unwrap();
        fftw::threading::plan_with_nthreads_f64(args.threads);

        if let Some(ref wisdom_path) = args.wisdom_file {
            let parent_path = wisdom_path.parent().unwrap();
            ensure_dir_exists(&parent_path);
            if wisdom_path.exists() {
                fftw::wisdom::import_wisdom_file_f64(&wisdom_path).unwrap();
            }
        }

        if let Some(ref image_path) = args.write_images {
            ensure_dir_exists(&image_path);
        }

        if let Some(ref dot_path) = args.write_dot {
            let parent_path = dot_path.parent().unwrap();
            ensure_dir_exists(&parent_path);
        }

        if let Some(ref timings_path) = args.timings_file {
            let parent_path = timings_path.parent().unwrap();
            ensure_dir_exists(&parent_path);
        }

        args
    }

    pub fn grid_bounds(&self) -> AABB<3> {
        let inclusive = self.domain_size as i32 - 1;
        AABB::new(matrix![0, inclusive; 0, inclusive; 0, inclusive])
    }

    pub fn frame_name(&self, i: usize) -> PathBuf {
        let mut result = self.write_images.as_ref().unwrap().clone();
        result.push(format!("frame_{i:04}.vtu"));
        result
    }

    pub fn finish(&self) {
        if let Some(ref wisdom_path) = self.wisdom_file {
            profiling::scope!("fftw3::saving_wisdom");
            println!("Saving wisdom: {wisdom_path:?}");
            fftw::wisdom::export_wisdom_file_f64(&wisdom_path).unwrap();
        }

        #[cfg(feature = "profile-with-puffin")]
        {
            println!("Flusing profiler");

            // We want to drop the server so we can flush the profiling data
            PUFFIN_SERVER.lock().unwrap().take();
        }
    }
}
