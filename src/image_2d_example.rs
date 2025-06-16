use crate::ap_solver::SolverParameters;
use crate::build_info;
use crate::domain::*;
use crate::fft_solver::PlanType;
use crate::image::image2d;
use crate::initial_conditions::*;
use crate::util::*;
use crate::SolverInterface;
use clap::Parser;
use std::path::PathBuf;
use std::time::*;

#[cfg(feature = "profile-with-puffin")]
use std::sync::Mutex;

#[cfg(feature = "profile-with-puffin")]
lazy_static::lazy_static! {
    static ref puffin_server: Mutex<Option<puffin_http::Server>> = {
        println!("Initializing profiling server:");
        let server_addr =
                format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
        println!(
                "Run this to view profiling data:  puffin_viewer {server_addr}"
            );
        let server = puffin_http::Server::new(&server_addr).unwrap();
        Mutex::new(Some(server))
    };
}

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

    /// Domain size, assume square
    #[arg(short, long, default_value = "1000")]
    pub domain_size: usize,

    /// Optionally write out images to this directory.
    /// WARNING: we do not check image size, so be reasonable.
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

    /// Fill with random values matching 2023 implementation
    #[arg(long)]
    pub ic_dial: Option<f64>,

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
}

impl Args {
    pub fn solver_parameters(&self) -> SolverParameters<2> {
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

    pub fn run_solver<SolverType: SolverInterface<2>>(
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

    pub fn run_solver_with_domains<'a, SolverType: SolverInterface<2>>(
        &self,
        input_domain: &mut SliceDomain<'a, 2>,
        output_domain: &mut SliceDomain<'a, 2>,
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
        generate_ic_2d(input_domain, ic_type, self.chunk_size);

        if self.write_images.is_some() {
            image2d(input_domain, &self.frame_name(0));
        }

        // Main solver loop
        let mut global_time = 0;
        for t in 1..self.images {
            // Run and time solver application
            let now = Instant::now();
            solver.apply(input_domain, output_domain, global_time);
            let elapsed_time = now.elapsed();
            eprintln!("{}", elapsed_time.as_nanos() as f64 / 1000000000.0);

            // Prepare for the next frame
            global_time += self.steps_per_image;
            std::mem::swap(input_domain, output_domain);

            // Write output (maybe)
            if self.write_images.is_some() {
                image2d(input_domain, &self.frame_name(t));
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
            let server_lock = &puffin_server.lock().unwrap();
            let server: &puffin_http::Server = server_lock.as_ref().unwrap();
            std::thread::sleep(std::time::Duration::from_secs(2));
            profiling::puffin::set_scopes_on(true);
            profiling::finish_frame!();
            println!("t: {}", &server.num_clients());
        }

        crate::init_threads(args.threads);

        if let Some(ref wisdom_path) = args.wisdom_file {
            if wisdom_path.exists() {
                fftw::wisdom::import_wisdom_file_f64(&wisdom_path).unwrap();
            }
        }

        args
    }

    pub fn grid_bounds(&self) -> AABB<2> {
        let inclusive = self.domain_size as i32 - 1;
        AABB::new(matrix![0, inclusive; 0, inclusive])
    }

    pub fn frame_name(&self, i: usize) -> PathBuf {
        let mut result = self.write_images.as_ref().unwrap().clone();
        result.push(format!("frame_{:04}.png", i));
        result
    }

    pub fn finish(&self) {
        if let Some(wisdom_path) = self.wisdom_file.as_ref() {
            profiling::scope!("fftw3::saving_wisdom");
            println!("Saving wisdom: {:?}", wisdom_path);
            fftw::wisdom::export_wisdom_file_f64(&wisdom_path).unwrap();
        }

        #[cfg(feature = "profile-with-puffin")]
        {
            println!("Flusing profiler");

            // We want to drop the server so we can flush the profiling data
            // https://stackoverflow.com/questions/68866598/how-do-i-free-memory-in-a-lazy-static
            puffin_server.lock().unwrap().take();
        }
    }
}
