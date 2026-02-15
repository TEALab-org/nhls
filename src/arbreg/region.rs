use std::collections::HashSet;

#[derive(Clone)]
pub struct Region {
    pub cells: HashSet<(i32, i32)>,
}

impl Region {
    pub fn empty() -> Self {
        Region {
            cells: HashSet::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn add(&mut self, x: i32, y: i32) {
        self.cells.insert((x, y));
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        self.cells.contains(&(x, y))
    }

    pub fn remove(&mut self, x: i32, y: i32) {
        self.cells.remove(&(x, y));
    }

    pub fn cell_iter(&self) -> impl Iterator<Item=&(i32, i32)> {
        self.cells.iter()
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    pub fn combine(&self, other: &Self) -> Self {
        let mut cells = self.cells.clone();
        cells.extend(other.cells.iter());
        Self {
            cells
        }
    }

    pub fn print_report(&self) {
        println!("BEGIN REGION REPORT:");
        println!(" - n cells: {}", self.cells.len());
        for (i, (x, y)) in self.cells.iter().enumerate() {
            println!(" - i: {}, x: {}, y: {}", i, x, y);
        }
    }
}
