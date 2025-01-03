use crate::domain::*;
use vtkio::model::*;

pub fn write_vtk3d<F: AsRef<std::path::Path>, DomainType: DomainView<3>>(
    domain: &DomainType,
    s: &F,
) {
    // Create inclusive range,
    // we need to reverse order of axis to match VTK's point ordering
    let aabb = domain.aabb();
    let bounds = aabb.bounds;
    let extent = Extent::Ranges([
        bounds[(2, 0)]..=bounds[(2, 1)],
        bounds[(1, 0)]..=bounds[(1, 1)],
        bounds[(0, 0)]..=bounds[(0, 1)],
    ]);

    let mut points = Vec::with_capacity(aabb.buffer_size() * 3);
    let exclusive_bounds = aabb.exclusive_bounds();
    for x in 0..exclusive_bounds[0] {
        for y in 0..exclusive_bounds[1] {
            for z in 0..exclusive_bounds[2] {
                points.push(z as f64);
                points.push(y as f64);
                points.push(x as f64);
            }
        }
    }
    println!("{:?}", points);

    let grid = StructuredGridPiece {
        extent,
        points: IOBuffer::F64(points),
        data: Attributes {
            point: vec![Attribute::DataArray(DataArray {
                name: "values".to_string(),
                elem: ElementType::Scalars {
                    num_comp: 1,
                    lookup_table: None,
                },
                data: IOBuffer::F64(domain.buffer().to_vec()),
            })],
            cell: vec![],
        },
    };

    Vtk {
        version: Version { major: 1, minor: 0 },
        title: String::new(),
        byte_order: ByteOrder::LittleEndian,
        file_path: None,
        data: DataSet::inline(grid),
    }
    .export(s)
    .unwrap();
}
