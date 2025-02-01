use crate::domain::*;
use nalgebra::vector;
use vtkio::model::*;

pub fn write_vtk3d<F: AsRef<std::path::Path>, DomainType: DomainView<3>>(
    domain: &DomainType,
    s: &F,
) {
    println!("Writing vtk: {:?}", s.as_ref());
    let aabb = domain.aabb();

    // Collect the grid points as vertices in mesh
    let buffer_size = aabb.buffer_size();
    let mut points = Vec::with_capacity(3 * buffer_size);
    for coord in aabb.coord_iter() {
        points.push(coord[0] as f32);
        points.push(coord[1] as f32);
        points.push(coord[2] as f32);
    }
    assert_eq!(points.len(), buffer_size * 3);

    // Assemble Hexahedron elements from grid points
    let cell_bounds = aabb.cell_bounds();
    let n_cells = cell_bounds.buffer_size();
    let mut connectivity = Vec::with_capacity(n_cells);
    let mut offsets = Vec::with_capacity(n_cells);
    let mut cell_types = Vec::with_capacity(n_cells);
    let mut offset = 8;
    for cell_coord in cell_bounds.coord_iter() {
        let n_1 = cell_coord + vector![0, 0, 1];
        let n_2 = cell_coord + vector![0, 1, 0];
        let n_3 = cell_coord + vector![1, 0, 0];
        let n_4 = cell_coord + vector![0, 1, 1];
        let n_5 = cell_coord + vector![1, 1, 0];
        let n_6 = cell_coord + vector![1, 0, 1];
        let n_7 = cell_coord + vector![1, 1, 1];

        let vertices = [&cell_coord, &n_3, &n_6, &n_1, &n_2, &n_5, &n_7, &n_4];
        for v in vertices {
            let index = aabb.coord_to_linear(v) as u64;
            connectivity.push(index);
        }

        offsets.push(offset);
        cell_types.push(CellType::Hexahedron);
        offset += 8;
    }

    let data: Vec<f32> = domain.buffer().iter().map(|v| *v as f32).collect();

    Vtk {
        version: Version::Auto,
        title: String::new(),
        byte_order: ByteOrder::LittleEndian,
        file_path: None,
        data: DataSet::inline(UnstructuredGridPiece {
            points: IOBuffer::F32(points),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity,
                    offsets,
                },
                types: cell_types,
            },
            data: Attributes {
                point: vec![Attribute::DataArray(DataArray {
                    name: "values".to_string(),
                    elem: ElementType::Scalars {
                        num_comp: 1,
                        lookup_table: None,
                    },
                    data: IOBuffer::F32(data),
                })],
                cell: vec![],
            },
        }),
    }
    .export(s)
    .unwrap();
}
