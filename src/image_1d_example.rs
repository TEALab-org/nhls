use crate::util::*;
use clap::Parser;
use std::path::PathBuf;

/// nhls 1D stencil executable
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Directory for output files, will be created.
    /// WARNING, if this Directory
    /// already exists, current contents will be removed.
    #[arg(short, long)]
    pub output_dir: std::path::PathBuf,

    /// Chunk size to use for parallelism.
    #[arg(short, long, default_value = "1000")]
    pub chunk_size: usize,

    /// How many lines the output image should have.
    #[arg(short, long, default_value = "1000")]
    pub lines: usize,

    /// How many steps to take per line.
    #[arg(short, long, default_value = "16")]
    pub steps_per_line: usize,

    /// Domain size
    #[arg(short, long, default_value = "1000")]
    pub domain_size: usize,
}

impl Args {
    pub fn cli_parse(name: &str) -> (Self, PathBuf) {
        println!("EXAMPLE: {}", name);
        println!("GIT DESCRIBE: {}", env!("GIT_DESCRIBE"));
        println!("GIT HASH: {}", env!("GIT_HASH"));
        let args = Args::parse();

        let output_dir = args.output_dir.to_str().unwrap();
        let _ = std::fs::remove_dir_all(output_dir);
        std::fs::create_dir(output_dir).unwrap();

        let mut output_image_path = args.output_dir.clone();
        output_image_path.push(format!("{}.png", name));

        (args, output_image_path)
    }

    pub fn grid_bounds(&self) -> AABB<1> {
        AABB::new(matrix![0, self.domain_size as i32 - 1])
    }
}
