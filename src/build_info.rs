pub fn print_report(name: &str) {
    println!("{{");
    println!("  \"name\": \"{}\",", name);
    println!("  \"git_describe\": \"{}\",", env!("GIT_DESCRIBE"));
    println!("  \"git_hash\": \"{}\"", env!("GIT_HASH"));
    println!("}}");
}
