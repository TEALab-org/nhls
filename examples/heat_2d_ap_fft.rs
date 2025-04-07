use nhls::domain::*;
use nhls::fft_solver::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init;
use std::time::*;

fn main() {
        #[cfg(feature = "profile-with-puffin")]
        let _s = {
            println!("Initializing profiling server:");
            let server_addr =
                format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
            let puffin_server =
                puffin_http::Server::new(&server_addr).unwrap();

            //while puffin_server.num_clients() < 1 {
            //    println!("No clients..., sleeping, {}", puffin_server.num_clients());
            //}

            profiling::puffin::set_scopes_on(true);
            profiling::finish_frame!();


            //std::thread::sleep(std::time::Duration::from_secs(2));
            println!(
                "Run this to view profiling data:  puffin_viewer {server_addr}"
            );
            println!("{}", puffin_server.num_clients());
            puffin_server
        };

    let args = Args::cli_setup("heat_2d_ap_fft");

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);
    let grid_bound = args.grid_bounds();

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Create AP Solver
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: args.cutoff,
        ratio: args.ratio,
        chunk_size: args.chunk_size,
    };
    let solver = APSolver::new(
        &bc,
        &stencil,
        grid_bound,
        args.steps_per_image,
        &planner_params,
        args.threads,
    );
    solver.print_report();

    if args.write_dot {
        solver.to_dot_file(&args.dot_path());
    }

    if args.gen_only {
        args.finish();
        std::process::exit(0);
    }

    // Create domains
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();

    if args.rand_init {
        init::rand(&mut input_domain, 1024, args.chunk_size);
    } else {
        init::normal_ic_2d(&mut input_domain, args.chunk_size);
    }

    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    let mut global_time = 0;
    for t in 1..args.images {
        let now = Instant::now();
        solver.apply(&mut input_domain, &mut output_domain, global_time);

        let elapsed_time = now.elapsed();
        eprintln!("{}", elapsed_time.as_nanos() as f64 / 1000000000.0);

        global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }

    args.finish();
}
