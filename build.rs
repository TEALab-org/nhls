// Get git info
// make available to src as constants
// https://stackoverflow.com/questions/43753491/include-git-commit-hash-as-string-into-rust-program
use std::process::Command;
fn main() {
    let describe_output = Command::new("git")
        .args(["describe", "--tags"])
        .output()
        .unwrap();
    let git_describe = String::from_utf8(describe_output.stdout).unwrap();
    println!("cargo:rustc-env=GIT_DESCRIBE={git_describe}");

    let hash_output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(hash_output.stdout).unwrap();
    println!("cargo:rustc-env=GIT_HASH={git_hash}");
}
