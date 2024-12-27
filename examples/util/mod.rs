use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    pub output_dir: std::path::PathBuf,
}

impl Args {
    pub fn cli_parse(name: &str) -> Self {
        println!("EXAMPLE: {}", name);
        println!("GIT: {}", env!("GIT_DESCRIBE"));
        let args = Args::parse();

        let output_dir = args.output_dir.to_str().unwrap();
        let _ = std::fs::remove_dir_all(output_dir);
        std::fs::create_dir(output_dir).unwrap();

        args
    }

    pub fn frame_name(&self, i: u32) -> std::path::PathBuf {
        let mut result = self.output_dir.to_path_buf();
        result.push(format!("frame_{:04}.png", i));
        result
    }
}
