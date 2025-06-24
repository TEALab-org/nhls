/// Utility function to ensure output directories exist when needed
pub fn ensure_dir_exists<P: AsRef<std::path::Path>>(path: &P) {
    // Check if it exists
    let p = path.as_ref();
    if p.exists() {
        if p.is_dir() {
            println!("Exists: {p:?}");
        } else {
            panic!("ERROR: not a directory {p:?}");
        }
    } else {
        println!("Creating: {p:?}");
        if let Err(e) = std::fs::create_dir_all(p) {
            panic!("ERROR: failed to create {p:?}, {e}");
        }
    }
}
