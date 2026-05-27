use flate2::read::ZlibDecoder;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MI_UTF8: u32 = 16;
const MI_UTF16: u32 = 17;
const MI_UTF32: u32 = 18;
const HDF5_SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const HDF5_SIGNATURE_SCAN_BYTES: usize = 1024 * 1024;
const HDF5_TEMP_COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const PARALLEL_MAT_WRITE_VALUES: usize = 100_000;

const MX_CHAR_CLASS: u32 = 4;
const MX_SPARSE_CLASS: u32 = 5;
const MX_DOUBLE_CLASS: u32 = 6;
const MX_SINGLE_CLASS: u32 = 7;
const MX_INT8_CLASS: u32 = 8;
const MX_UINT8_CLASS: u32 = 9;
const MX_INT16_CLASS: u32 = 10;
const MX_UINT16_CLASS: u32 = 11;
const MX_INT32_CLASS: u32 = 12;
const MX_UINT32_CLASS: u32 = 13;
const MX_INT64_CLASS: u32 = 14;
const MX_UINT64_CLASS: u32 = 15;
const MX_COMPLEX_FLAG: u32 = 0x0800;
const MX_LOGICAL_FLAG: u32 = 0x0200;

#[derive(Debug, Error)]
pub enum MatError {
    #[error("unable to read MAT file {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("unable to write MAT file {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("MAT file {path} has an invalid v5 header: {message}")]
    InvalidHeader { path: PathBuf, message: String },
    #[error("MAT file {path} contains unsupported endian marker {marker:?}")]
    UnsupportedEndian { path: PathBuf, marker: [u8; 2] },
    #[error("MAT variable {name} has {actual} values for {rows}x{cols}")]
    Shape {
        name: String,
        rows: usize,
        cols: usize,
        actual: usize,
    },
    #[error("MAT variable {name} has unsupported class {class}")]
    UnsupportedClass { name: String, class: u32 },
    #[error("MAT variable {name} has unsupported data type {data_type}")]
    UnsupportedDataType { name: String, data_type: u32 },
    #[error(
        "MAT variable {name} has malformed dimensions {dims:?}; expected exactly 2 dimensions"
    )]
    MalformedDimensions { name: String, dims: Vec<i32> },
    #[error("MAT variable {name} is malformed: {message}")]
    MalformedVariable { name: String, message: String },
    #[error("MAT variable {name} is missing")]
    MissingVariable { name: String },
    #[error("MAT variable {name} has type {actual}; expected {expected}")]
    TypeMismatch {
        name: String,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("MAT file {path} is malformed: {message}")]
    MalformedFile { path: PathBuf, message: String },
}

pub struct MatFile {
    path: PathBuf,
    variables: Vec<MatVar>,
}

#[derive(Clone, Debug)]
enum MatVar {
    F64(Matrix<f64>),
    F32(Matrix<f32>),
    F32ColumnMajor(Matrix<f32>),
    I32(Matrix<i32>),
    U32(Matrix<u32>),
    U8(Matrix<u8>),
    ComplexF64(ComplexMatrixF64),
    ComplexF32(ComplexMatrixF32),
    ComplexF32Array(ComplexArrayF32),
    SparseCscF64(SparseCscMatrixF64),
}

#[derive(Clone, Debug)]
pub struct Matrix<T> {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<T>,
}

#[derive(Clone, Debug)]
pub struct ComplexMatrixF32 {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<(f32, f32)>,
}

#[derive(Clone, Debug)]
pub struct ComplexMatrixF64 {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<(f64, f64)>,
}

#[derive(Clone, Debug)]
struct ComplexArrayF32 {
    name: String,
    dims: Vec<usize>,
    values: Vec<(f32, f32)>,
}

#[derive(Clone, Debug)]
pub struct SparseCscMatrixF64 {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
    pub ir: Vec<i32>,
    pub jc: Vec<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericType {
    F64,
    F32,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NumericData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatArray {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub numeric_type: NumericType,
    pub real: NumericData,
    pub imag: Option<NumericData>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatData {
    variables: BTreeMap<String, MatArray>,
}

pub const VAR_IJ: &str = "ij";
pub const VAR_LONLAT: &str = "lonlat";
pub const VAR_XY: &str = "xy";
pub const VAR_BPERP: &str = "bperp";
pub const VAR_BPERP_MAT: &str = "bperp_mat";
pub const VAR_DAY: &str = "day";
pub const VAR_MASTER_DAY: &str = "master_day";
pub const VAR_MASTER_IX: &str = "master_ix";
pub const VAR_N_IFG: &str = "n_ifg";
pub const VAR_N_IMAGE: &str = "n_image";
pub const VAR_N_PS: &str = "n_ps";
pub const VAR_SORT_IX: &str = "sort_ix";
pub const VAR_LL0: &str = "ll0";
pub const VAR_MEAN_RANGE: &str = "mean_range";
pub const VAR_MEAN_INCIDENCE: &str = "mean_incidence";
pub const VAR_PH: &str = "ph";
pub const VAR_PSVER: &str = "psver";
pub const VAR_D_A: &str = "D_A";
pub const VAR_HGT: &str = "hgt";
pub const VAR_LA: &str = "la";

#[derive(Clone, Debug)]
pub struct Ps1Artifact {
    pub ij: Matrix<f64>,
    pub lonlat: Matrix<f64>,
    pub xy: Matrix<f32>,
    pub bperp: Matrix<f64>,
    pub day: Matrix<f64>,
    pub master_day: f64,
    pub master_ix: f64,
    pub n_ifg: f64,
    pub n_image: f64,
    pub n_ps: f64,
    pub sort_ix: Matrix<f64>,
    pub ll0: Matrix<f64>,
    pub mean_range: f64,
    pub mean_incidence: f64,
}

impl MatFile {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            variables: Vec::new(),
        }
    }

    pub fn read(path: impl AsRef<Path>) -> Result<MatData, MatError> {
        MatData::read(path)
    }

    pub fn add_f64_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        values: Vec<f64>,
    ) -> Result<(), MatError> {
        let matrix = matrix_with_values(name.into(), rows, cols, values)?;
        self.variables.push(MatVar::F64(matrix));
        Ok(())
    }

    pub fn add_f64_scalar(&mut self, name: impl Into<String>, value: f64) -> Result<(), MatError> {
        self.add_f64_matrix(name, 1, 1, vec![value])
    }

    pub fn add_f64_row_vector(
        &mut self,
        name: impl Into<String>,
        values: Vec<f64>,
    ) -> Result<(), MatError> {
        self.add_f64_matrix(name, 1, values.len(), values)
    }

    pub fn add_f64_col_vector(
        &mut self,
        name: impl Into<String>,
        values: Vec<f64>,
    ) -> Result<(), MatError> {
        self.add_f64_matrix(name, values.len(), 1, values)
    }

    pub fn add_f32_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        values: Vec<f32>,
    ) -> Result<(), MatError> {
        let matrix = matrix_with_values(name.into(), rows, cols, values)?;
        self.variables.push(MatVar::F32(matrix));
        Ok(())
    }

    pub fn add_f32_column_major_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        values: Vec<f32>,
    ) -> Result<(), MatError> {
        let matrix = matrix_with_values(name.into(), rows, cols, values)?;
        self.variables.push(MatVar::F32ColumnMajor(matrix));
        Ok(())
    }

    pub fn add_f32_scalar(&mut self, name: impl Into<String>, value: f32) -> Result<(), MatError> {
        self.add_f32_matrix(name, 1, 1, vec![value])
    }

    pub fn add_f32_row_vector(
        &mut self,
        name: impl Into<String>,
        values: Vec<f32>,
    ) -> Result<(), MatError> {
        self.add_f32_matrix(name, 1, values.len(), values)
    }

    pub fn add_f32_col_vector(
        &mut self,
        name: impl Into<String>,
        values: Vec<f32>,
    ) -> Result<(), MatError> {
        self.add_f32_matrix(name, values.len(), 1, values)
    }

    pub fn add_i32_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        values: Vec<i32>,
    ) -> Result<(), MatError> {
        let matrix = matrix_with_values(name.into(), rows, cols, values)?;
        self.variables.push(MatVar::I32(matrix));
        Ok(())
    }

    pub fn add_u32_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        values: Vec<u32>,
    ) -> Result<(), MatError> {
        let matrix = matrix_with_values(name.into(), rows, cols, values)?;
        self.variables.push(MatVar::U32(matrix));
        Ok(())
    }

    pub fn add_u8_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        values: Vec<u8>,
    ) -> Result<(), MatError> {
        let matrix = matrix_with_values(name.into(), rows, cols, values)?;
        self.variables.push(MatVar::U8(matrix));
        Ok(())
    }

    pub fn add_complex_f32_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        values: Vec<(f32, f32)>,
    ) -> Result<(), MatError> {
        let matrix = complex_f32_with_values(name.into(), rows, cols, values)?;
        self.variables.push(MatVar::ComplexF32(matrix));
        Ok(())
    }

    pub fn add_complex_f32_array3(
        &mut self,
        name: impl Into<String>,
        dim0: usize,
        dim1: usize,
        dim2: usize,
        values: Vec<(f32, f32)>,
    ) -> Result<(), MatError> {
        let name = name.into();
        let Some(expected) = dim0
            .checked_mul(dim1)
            .and_then(|value| value.checked_mul(dim2))
        else {
            return Err(MatError::Shape {
                name,
                rows: dim0,
                cols: dim1.saturating_mul(dim2),
                actual: values.len(),
            });
        };
        if values.len() != expected {
            return Err(MatError::Shape {
                name,
                rows: dim0,
                cols: dim1.saturating_mul(dim2),
                actual: values.len(),
            });
        }
        self.variables
            .push(MatVar::ComplexF32Array(ComplexArrayF32 {
                name,
                dims: vec![dim0, dim1, dim2],
                values,
            }));
        Ok(())
    }

    pub fn add_complex_f64_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        values: Vec<(f64, f64)>,
    ) -> Result<(), MatError> {
        let matrix = complex_f64_with_values(name.into(), rows, cols, values)?;
        self.variables.push(MatVar::ComplexF64(matrix));
        Ok(())
    }

    pub fn add_sparse_csc_f64_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        data: Vec<f64>,
        ir: Vec<i32>,
        jc: Vec<i32>,
    ) -> Result<(), MatError> {
        let matrix = sparse_csc_f64_with_values(name.into(), rows, cols, data, ir, jc)?;
        self.variables.push(MatVar::SparseCscF64(matrix));
        Ok(())
    }

    pub fn add_empty_sparse_csc_f64_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
    ) -> Result<(), MatError> {
        self.add_sparse_csc_f64_matrix(name, rows, cols, Vec::new(), Vec::new(), vec![0; cols + 1])
    }

    pub fn write(&self) -> Result<(), MatError> {
        let mut file = File::create(&self.path).map_err(|source| MatError::Write {
            path: self.path.clone(),
            source,
        })?;
        write_header(&mut file).map_err(|source| MatError::Write {
            path: self.path.clone(),
            source,
        })?;
        for variable in &self.variables {
            write_variable(&mut file, variable).map_err(|source| MatError::Write {
                path: self.path.clone(),
                source,
            })?;
        }
        Ok(())
    }
}

impl MatData {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, MatError> {
        Self::read_with_vars(path.as_ref(), None)
    }

    pub fn read_selected(path: impl AsRef<Path>, variables: &[&str]) -> Result<Self, MatError> {
        Self::read_with_vars(path.as_ref(), Some(variables))
    }

    fn read_with_vars(path: &Path, variables: Option<&[&str]>) -> Result<Self, MatError> {
        let mut probe = File::open(path).map_err(|source| MatError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let mut bytes = vec![0_u8; HDF5_SIGNATURE_SCAN_BYTES];
        let bytes_read = probe.read(&mut bytes).map_err(|source| MatError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        bytes.truncate(bytes_read);
        if let Some(offset) = find_hdf5_signature_offset(&bytes) {
            return read_hdf5_mat_file(path, offset, variables);
        }

        let bytes = std::fs::read(path).map_err(|source| MatError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        parse_mat_file(path, &bytes, variables)
    }

    pub fn variables(&self) -> impl Iterator<Item = (&str, &MatArray)> {
        self.variables
            .iter()
            .map(|(name, array)| (name.as_str(), array))
    }

    pub fn get(&self, name: &str) -> Result<&MatArray, MatError> {
        self.variables
            .get(name)
            .ok_or_else(|| MatError::MissingVariable {
                name: name.to_string(),
            })
    }

    pub fn get_f64_matrix(&self, name: &str) -> Result<Matrix<f64>, MatError> {
        self.get(name)?.to_f64_matrix()
    }

    pub fn get_f32_matrix(&self, name: &str) -> Result<Matrix<f32>, MatError> {
        self.get(name)?.to_f32_matrix()
    }

    pub fn get_complex_f32_matrix(&self, name: &str) -> Result<ComplexMatrixF32, MatError> {
        self.get(name)?.to_complex_f32_matrix()
    }
}

pub fn read_hdf5_f32_dataset_raw(
    path: impl AsRef<Path>,
    name: &str,
) -> Result<Matrix<f32>, MatError> {
    read_hdf5_f32_datasets_raw(path, &[name]).and_then(|mut values| {
        values
            .remove(name)
            .ok_or_else(|| MatError::MissingVariable {
                name: name.to_string(),
            })
    })
}

pub fn read_hdf5_f32_datasets_raw(
    path: impl AsRef<Path>,
    names: &[&str],
) -> Result<BTreeMap<String, Matrix<f32>>, MatError> {
    let path = path.as_ref();
    let file = rust_hdf5::H5File::open(path).map_err(|err| MatError::MalformedFile {
        path: path.to_path_buf(),
        message: format!("unable to open HDF5 MAT payload: {err}"),
    })?;
    let mut out = BTreeMap::new();
    for &name in names {
        let dataset = file.dataset(name).map_err(|_| MatError::MissingVariable {
            name: name.to_string(),
        })?;
        let values = dataset
            .read_raw::<f32>()
            .map_err(|err| MatError::UnsupportedDataType {
                name: format!("HDF5 dataset {name} in {} ({err})", path.display()),
                data_type: 0,
            })?;
        let shape = dataset.shape();
        let (rows, cols) = match shape.as_slice() {
            [] => (1, 1),
            [_] => (values.len(), 1),
            [rows, cols] => (*rows, *cols),
            _ => {
                let rows = *shape.first().unwrap_or(&1);
                let cols = values.len() / rows.max(1);
                (rows, cols)
            }
        };
        out.insert(
            name.to_string(),
            matrix_with_values(name.to_string(), rows, cols, values)?,
        );
    }
    Ok(out)
}

impl MatArray {
    pub fn is_complex(&self) -> bool {
        self.imag.is_some()
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_scalar(&self) -> bool {
        self.rows == 1 && self.cols == 1
    }

    pub fn is_row_vector(&self) -> bool {
        self.rows == 1
    }

    pub fn is_col_vector(&self) -> bool {
        self.cols == 1
    }

    pub fn to_f64_matrix(&self) -> Result<Matrix<f64>, MatError> {
        if self.is_complex() {
            return Err(MatError::TypeMismatch {
                name: self.name.clone(),
                expected: "real numeric matrix",
                actual: "complex matrix",
            });
        }
        Ok(Matrix {
            name: self.name.clone(),
            rows: self.rows,
            cols: self.cols,
            values: self.real.to_f64_vec(),
        })
    }

    pub fn to_f32_matrix(&self) -> Result<Matrix<f32>, MatError> {
        if self.is_complex() {
            return Err(MatError::TypeMismatch {
                name: self.name.clone(),
                expected: "real numeric matrix",
                actual: "complex matrix",
            });
        }
        Ok(Matrix {
            name: self.name.clone(),
            rows: self.rows,
            cols: self.cols,
            values: self.real.to_f32_vec(),
        })
    }

    pub fn to_complex_f32_matrix(&self) -> Result<ComplexMatrixF32, MatError> {
        let Some(imag) = &self.imag else {
            return Err(MatError::TypeMismatch {
                name: self.name.clone(),
                expected: "complex matrix",
                actual: "real numeric matrix",
            });
        };
        let real = self.real.to_f32_vec();
        let imag = imag.to_f32_vec();
        if real.len() != imag.len() {
            return Err(MatError::MalformedVariable {
                name: self.name.clone(),
                message: "real and imaginary payload lengths differ".to_string(),
            });
        }
        Ok(ComplexMatrixF32 {
            name: self.name.clone(),
            rows: self.rows,
            cols: self.cols,
            values: real.into_iter().zip(imag).collect(),
        })
    }
}

impl NumericData {
    pub fn numeric_type(&self) -> NumericType {
        match self {
            NumericData::F64(_) => NumericType::F64,
            NumericData::F32(_) => NumericType::F32,
            NumericData::I8(_) => NumericType::I8,
            NumericData::U8(_) => NumericType::U8,
            NumericData::I16(_) => NumericType::I16,
            NumericData::U16(_) => NumericType::U16,
            NumericData::I32(_) => NumericType::I32,
            NumericData::U32(_) => NumericType::U32,
            NumericData::I64(_) => NumericType::I64,
            NumericData::U64(_) => NumericType::U64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            NumericData::F64(values) => values.len(),
            NumericData::F32(values) => values.len(),
            NumericData::I8(values) => values.len(),
            NumericData::U8(values) => values.len(),
            NumericData::I16(values) => values.len(),
            NumericData::U16(values) => values.len(),
            NumericData::I32(values) => values.len(),
            NumericData::U32(values) => values.len(),
            NumericData::I64(values) => values.len(),
            NumericData::U64(values) => values.len(),
        }
    }

    fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            NumericData::F64(values) => values.clone(),
            NumericData::F32(values) => values.iter().map(|&value| value as f64).collect(),
            NumericData::I8(values) => values.iter().map(|&value| value as f64).collect(),
            NumericData::U8(values) => values.iter().map(|&value| value as f64).collect(),
            NumericData::I16(values) => values.iter().map(|&value| value as f64).collect(),
            NumericData::U16(values) => values.iter().map(|&value| value as f64).collect(),
            NumericData::I32(values) => values.iter().map(|&value| value as f64).collect(),
            NumericData::U32(values) => values.iter().map(|&value| value as f64).collect(),
            NumericData::I64(values) => values.iter().map(|&value| value as f64).collect(),
            NumericData::U64(values) => values.iter().map(|&value| value as f64).collect(),
        }
    }

    fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            NumericData::F64(values) => values.iter().map(|&value| value as f32).collect(),
            NumericData::F32(values) => values.clone(),
            NumericData::I8(values) => values.iter().map(|&value| value as f32).collect(),
            NumericData::U8(values) => values.iter().map(|&value| value as f32).collect(),
            NumericData::I16(values) => values.iter().map(|&value| value as f32).collect(),
            NumericData::U16(values) => values.iter().map(|&value| value as f32).collect(),
            NumericData::I32(values) => values.iter().map(|&value| value as f32).collect(),
            NumericData::U32(values) => values.iter().map(|&value| value as f32).collect(),
            NumericData::I64(values) => values.iter().map(|&value| value as f32).collect(),
            NumericData::U64(values) => values.iter().map(|&value| value as f32).collect(),
        }
    }
}

impl Ps1Artifact {
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), MatError> {
        let mut mat = MatFile::new(path);
        mat.add_f64_matrix(VAR_IJ, self.ij.rows, self.ij.cols, self.ij.values.clone())?;
        mat.add_f64_matrix(
            VAR_LONLAT,
            self.lonlat.rows,
            self.lonlat.cols,
            self.lonlat.values.clone(),
        )?;
        mat.add_f32_matrix(VAR_XY, self.xy.rows, self.xy.cols, self.xy.values.clone())?;
        mat.add_f64_matrix(
            VAR_BPERP,
            self.bperp.rows,
            self.bperp.cols,
            self.bperp.values.clone(),
        )?;
        mat.add_f64_matrix(
            VAR_DAY,
            self.day.rows,
            self.day.cols,
            self.day.values.clone(),
        )?;
        mat.add_f64_scalar(VAR_MASTER_DAY, self.master_day)?;
        mat.add_f64_scalar(VAR_MASTER_IX, self.master_ix)?;
        mat.add_f64_scalar(VAR_N_IFG, self.n_ifg)?;
        mat.add_f64_scalar(VAR_N_IMAGE, self.n_image)?;
        mat.add_f64_scalar(VAR_N_PS, self.n_ps)?;
        mat.add_f64_matrix(
            VAR_SORT_IX,
            self.sort_ix.rows,
            self.sort_ix.cols,
            self.sort_ix.values.clone(),
        )?;
        mat.add_f64_matrix(
            VAR_LL0,
            self.ll0.rows,
            self.ll0.cols,
            self.ll0.values.clone(),
        )?;
        mat.add_f64_scalar(VAR_MEAN_RANGE, self.mean_range)?;
        mat.add_f64_scalar(VAR_MEAN_INCIDENCE, self.mean_incidence)?;
        mat.write()
    }
}

pub fn f64_matrix(
    name: impl Into<String>,
    rows: usize,
    cols: usize,
    values: Vec<f64>,
) -> Result<Matrix<f64>, MatError> {
    matrix_with_values(name.into(), rows, cols, values)
}

pub fn f32_matrix(
    name: impl Into<String>,
    rows: usize,
    cols: usize,
    values: Vec<f32>,
) -> Result<Matrix<f32>, MatError> {
    matrix_with_values(name.into(), rows, cols, values)
}

pub fn write_phase_artifact(
    path: impl AsRef<Path>,
    rows: usize,
    cols: usize,
    ph: Vec<(f32, f32)>,
) -> Result<(), MatError> {
    let mut mat = MatFile::new(path);
    mat.add_complex_f32_matrix(VAR_PH, rows, cols, ph)?;
    mat.write()
}

pub fn write_baseline_artifact(
    path: impl AsRef<Path>,
    rows: usize,
    cols: usize,
    bperp_mat: Vec<f32>,
) -> Result<(), MatError> {
    let mut mat = MatFile::new(path);
    mat.add_f32_matrix(VAR_BPERP_MAT, rows, cols, bperp_mat)?;
    mat.write()
}

pub fn write_psver_artifact(path: impl AsRef<Path>, version: f64) -> Result<(), MatError> {
    let mut mat = MatFile::new(path);
    mat.add_f64_scalar(VAR_PSVER, version)?;
    mat.write()
}

pub fn write_da_artifact(path: impl AsRef<Path>, values: Vec<f64>) -> Result<(), MatError> {
    let mut mat = MatFile::new(path);
    mat.add_f64_row_vector(VAR_D_A, values)?;
    mat.write()
}

pub fn write_hgt_artifact(path: impl AsRef<Path>, values: Vec<f32>) -> Result<(), MatError> {
    let mut mat = MatFile::new(path);
    mat.add_f32_row_vector(VAR_HGT, values)?;
    mat.write()
}

pub fn write_la_artifact(path: impl AsRef<Path>, values: Vec<f64>) -> Result<(), MatError> {
    let mut mat = MatFile::new(path);
    mat.add_f64_row_vector(VAR_LA, values)?;
    mat.write()
}

pub fn read_phase_artifact(path: impl AsRef<Path>) -> Result<ComplexMatrixF32, MatError> {
    MatData::read(path)?.get_complex_f32_matrix(VAR_PH)
}

pub fn read_baseline_artifact(path: impl AsRef<Path>) -> Result<Matrix<f32>, MatError> {
    MatData::read(path)?.get_f32_matrix(VAR_BPERP_MAT)
}

fn matrix_with_values<T>(
    name: String,
    rows: usize,
    cols: usize,
    values: Vec<T>,
) -> Result<Matrix<T>, MatError> {
    let Some(expected) = rows.checked_mul(cols) else {
        return Err(MatError::Shape {
            name,
            rows,
            cols,
            actual: values.len(),
        });
    };
    if values.len() != expected {
        return Err(MatError::Shape {
            name,
            rows,
            cols,
            actual: values.len(),
        });
    }
    Ok(Matrix {
        name,
        rows,
        cols,
        values,
    })
}

fn complex_f32_with_values(
    name: String,
    rows: usize,
    cols: usize,
    values: Vec<(f32, f32)>,
) -> Result<ComplexMatrixF32, MatError> {
    let Some(expected) = rows.checked_mul(cols) else {
        return Err(MatError::Shape {
            name,
            rows,
            cols,
            actual: values.len(),
        });
    };
    if values.len() != expected {
        return Err(MatError::Shape {
            name,
            rows,
            cols,
            actual: values.len(),
        });
    }
    Ok(ComplexMatrixF32 {
        name,
        rows,
        cols,
        values,
    })
}

fn complex_f64_with_values(
    name: String,
    rows: usize,
    cols: usize,
    values: Vec<(f64, f64)>,
) -> Result<ComplexMatrixF64, MatError> {
    let Some(expected) = rows.checked_mul(cols) else {
        return Err(MatError::Shape {
            name,
            rows,
            cols,
            actual: values.len(),
        });
    };
    if values.len() != expected {
        return Err(MatError::Shape {
            name,
            rows,
            cols,
            actual: values.len(),
        });
    }
    Ok(ComplexMatrixF64 {
        name,
        rows,
        cols,
        values,
    })
}

fn sparse_csc_f64_with_values(
    name: String,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
    ir: Vec<i32>,
    jc: Vec<i32>,
) -> Result<SparseCscMatrixF64, MatError> {
    if data.len() != ir.len() {
        return Err(MatError::Shape {
            name,
            rows,
            cols,
            actual: data.len().max(ir.len()),
        });
    }
    if jc.len() != cols + 1 {
        return Err(MatError::MalformedVariable {
            name,
            message: format!(
                "sparse jc length {} does not match cols + 1 ({})",
                jc.len(),
                cols + 1
            ),
        });
    }
    if jc.first().copied().unwrap_or_default() != 0 {
        return Err(MatError::MalformedVariable {
            name,
            message: "sparse jc must start at 0".to_string(),
        });
    }
    if jc.last().copied().unwrap_or_default() != data.len() as i32 {
        return Err(MatError::MalformedVariable {
            name,
            message: "sparse jc final pointer must equal data length".to_string(),
        });
    }
    if jc.windows(2).any(|window| window[0] > window[1]) {
        return Err(MatError::MalformedVariable {
            name,
            message: "sparse jc must be monotonically nondecreasing".to_string(),
        });
    }
    if ir.iter().any(|&row| row < 0 || row as usize >= rows) {
        return Err(MatError::MalformedVariable {
            name,
            message: "sparse ir contains an out-of-range row index".to_string(),
        });
    }
    Ok(SparseCscMatrixF64 {
        name,
        rows,
        cols,
        data,
        ir,
        jc,
    })
}

fn write_header(file: &mut File) -> io::Result<()> {
    let mut text = [b' '; 116];
    let description = b"MATLAB 5.0 MAT-file, Platform: pySTAMPS Rust native";
    text[..description.len()].copy_from_slice(description);
    file.write_all(&text)?;
    file.write_all(&[0; 8])?;
    file.write_all(&0x0100u16.to_le_bytes())?;
    file.write_all(b"IM")?;
    Ok(())
}

fn write_variable(file: &mut File, variable: &MatVar) -> io::Result<()> {
    let mut body = Vec::new();
    match variable {
        MatVar::F64(matrix) => write_real_f64_matrix(&mut body, matrix)?,
        MatVar::F32(matrix) => write_real_f32_matrix(&mut body, matrix)?,
        MatVar::F32ColumnMajor(matrix) => write_real_f32_column_major_matrix(&mut body, matrix)?,
        MatVar::I32(matrix) => write_real_i32_matrix(&mut body, matrix)?,
        MatVar::U32(matrix) => write_real_u32_matrix(&mut body, matrix)?,
        MatVar::U8(matrix) => write_real_u8_matrix(&mut body, matrix)?,
        MatVar::ComplexF32(matrix) => {
            write_array_flags(&mut body, MX_SINGLE_CLASS, true)?;
            write_dimensions(&mut body, matrix.rows, matrix.cols)?;
            write_name(&mut body, &matrix.name)?;
            write_complex_f32(matrix, &mut body)?;
        }
        MatVar::ComplexF32Array(array) => {
            write_array_flags(&mut body, MX_SINGLE_CLASS, true)?;
            write_dimensions_nd(&mut body, &array.dims)?;
            write_name(&mut body, &array.name)?;
            write_complex_f32_array(array, &mut body)?;
        }
        MatVar::ComplexF64(matrix) => {
            write_array_flags(&mut body, MX_DOUBLE_CLASS, true)?;
            write_dimensions(&mut body, matrix.rows, matrix.cols)?;
            write_name(&mut body, &matrix.name)?;
            write_complex_f64(matrix, &mut body)?;
        }
        MatVar::SparseCscF64(matrix) => {
            write_sparse_array_flags(&mut body, matrix.data.len())?;
            write_dimensions(&mut body, matrix.rows, matrix.cols)?;
            write_name(&mut body, &matrix.name)?;
            write_sparse_i32_data(&mut body, &matrix.ir)?;
            write_sparse_i32_data(&mut body, &matrix.jc)?;
            write_sparse_f64_data(&mut body, &matrix.data)?;
        }
    }
    write_tag(file, MI_MATRIX, body.len())?;
    file.write_all(&body)?;
    pad_to_8(file, body.len())
}

fn write_real_header<T>(out: &mut Vec<u8>, class: u32, matrix: &Matrix<T>) -> io::Result<()> {
    write_array_flags(out, class, false)?;
    write_dimensions(out, matrix.rows, matrix.cols)?;
    write_name(out, &matrix.name)
}

fn write_real_f64_matrix(out: &mut Vec<u8>, matrix: &Matrix<f64>) -> io::Result<()> {
    write_real_header(out, MX_DOUBLE_CLASS, matrix)?;
    write_numeric_data_f64(out, MI_DOUBLE, matrix.rows, matrix.cols, &matrix.values)
}

fn write_real_f32_matrix(out: &mut Vec<u8>, matrix: &Matrix<f32>) -> io::Result<()> {
    write_real_header(out, MX_SINGLE_CLASS, matrix)?;
    write_numeric_data_f32(out, MI_SINGLE, matrix.rows, matrix.cols, &matrix.values)
}

fn write_real_f32_column_major_matrix(out: &mut Vec<u8>, matrix: &Matrix<f32>) -> io::Result<()> {
    write_real_header(out, MX_SINGLE_CLASS, matrix)?;
    write_numeric_data_f32_column_major(out, MI_SINGLE, &matrix.values)
}

fn write_real_i32_matrix(out: &mut Vec<u8>, matrix: &Matrix<i32>) -> io::Result<()> {
    write_real_header(out, MX_INT32_CLASS, matrix)?;
    write_numeric_data_i32(out, MI_INT32, matrix.rows, matrix.cols, &matrix.values)
}

fn write_real_u32_matrix(out: &mut Vec<u8>, matrix: &Matrix<u32>) -> io::Result<()> {
    write_real_header(out, MX_UINT32_CLASS, matrix)?;
    write_numeric_data_u32(out, MI_UINT32, matrix.rows, matrix.cols, &matrix.values)
}

fn write_real_u8_matrix(out: &mut Vec<u8>, matrix: &Matrix<u8>) -> io::Result<()> {
    write_real_header(out, MX_UINT8_CLASS, matrix)?;
    write_numeric_data_u8(out, MI_UINT8, matrix.rows, matrix.cols, &matrix.values)
}

fn write_array_flags(out: &mut Vec<u8>, class: u32, complex: bool) -> io::Result<()> {
    write_tag(out, MI_UINT32, 8)?;
    let flags = if complex {
        class | MX_COMPLEX_FLAG
    } else {
        class
    };
    out.write_all(&flags.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())
}

fn write_sparse_array_flags(out: &mut Vec<u8>, nnz: usize) -> io::Result<()> {
    write_tag(out, MI_UINT32, 8)?;
    out.write_all(&MX_SPARSE_CLASS.to_le_bytes())?;
    out.write_all(&(nnz as u32).to_le_bytes())
}

fn write_dimensions(out: &mut Vec<u8>, rows: usize, cols: usize) -> io::Result<()> {
    write_tag(out, MI_INT32, 8)?;
    out.write_all(&(rows as i32).to_le_bytes())?;
    out.write_all(&(cols as i32).to_le_bytes())
}

fn write_dimensions_nd(out: &mut Vec<u8>, dims: &[usize]) -> io::Result<()> {
    let byte_len = dims.len() * std::mem::size_of::<i32>();
    write_tag(out, MI_INT32, byte_len)?;
    for &dim in dims {
        out.write_all(&(dim as i32).to_le_bytes())?;
    }
    pad_to_8(out, byte_len)
}

fn write_name(out: &mut Vec<u8>, name: &str) -> io::Result<()> {
    write_tag(out, MI_INT8, name.len())?;
    out.write_all(name.as_bytes())?;
    pad_to_8(out, name.len())
}

fn write_numeric_data_f64(
    out: &mut Vec<u8>,
    data_type: u32,
    rows: usize,
    cols: usize,
    values: &[f64],
) -> io::Result<()> {
    let byte_len = std::mem::size_of_val(values);
    write_tag(out, data_type, byte_len)?;
    let start = out.len();
    out.resize(start + byte_len, 0);
    let payload = &mut out[start..start + byte_len];
    if values.len() >= PARALLEL_MAT_WRITE_VALUES {
        payload
            .par_chunks_mut(std::mem::size_of::<f64>())
            .enumerate()
            .for_each(|(dst, chunk)| {
                let col = dst / rows;
                let row = dst % rows;
                chunk.copy_from_slice(&values[row * cols + col].to_le_bytes());
            });
    } else {
        for (dst, chunk) in payload.chunks_mut(std::mem::size_of::<f64>()).enumerate() {
            let col = dst / rows;
            let row = dst % rows;
            chunk.copy_from_slice(&values[row * cols + col].to_le_bytes());
        }
    }
    pad_to_8(out, byte_len)
}

fn write_numeric_data_f32(
    out: &mut Vec<u8>,
    data_type: u32,
    rows: usize,
    cols: usize,
    values: &[f32],
) -> io::Result<()> {
    let byte_len = std::mem::size_of_val(values);
    write_tag(out, data_type, byte_len)?;
    let start = out.len();
    out.resize(start + byte_len, 0);
    let payload = &mut out[start..start + byte_len];
    if values.len() >= PARALLEL_MAT_WRITE_VALUES {
        payload
            .par_chunks_mut(std::mem::size_of::<f32>())
            .enumerate()
            .for_each(|(dst, chunk)| {
                let col = dst / rows;
                let row = dst % rows;
                chunk.copy_from_slice(&values[row * cols + col].to_le_bytes());
            });
    } else {
        for (dst, chunk) in payload.chunks_mut(std::mem::size_of::<f32>()).enumerate() {
            let col = dst / rows;
            let row = dst % rows;
            chunk.copy_from_slice(&values[row * cols + col].to_le_bytes());
        }
    }
    pad_to_8(out, byte_len)
}

fn write_numeric_data_f32_column_major(
    out: &mut Vec<u8>,
    data_type: u32,
    values: &[f32],
) -> io::Result<()> {
    let byte_len = std::mem::size_of_val(values);
    write_tag(out, data_type, byte_len)?;
    let start = out.len();
    out.resize(start + byte_len, 0);
    let payload = &mut out[start..start + byte_len];
    if values.len() >= PARALLEL_MAT_WRITE_VALUES {
        payload
            .par_chunks_mut(std::mem::size_of::<f32>())
            .zip(values.par_iter())
            .for_each(|(chunk, value)| {
                chunk.copy_from_slice(&value.to_le_bytes());
            });
    } else {
        for (chunk, value) in payload
            .chunks_mut(std::mem::size_of::<f32>())
            .zip(values.iter())
        {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
    }
    pad_to_8(out, byte_len)
}

fn write_numeric_data_i32(
    out: &mut Vec<u8>,
    data_type: u32,
    rows: usize,
    cols: usize,
    values: &[i32],
) -> io::Result<()> {
    let byte_len = std::mem::size_of_val(values);
    write_tag(out, data_type, byte_len)?;
    let start = out.len();
    out.resize(start + byte_len, 0);
    let payload = &mut out[start..start + byte_len];
    if values.len() >= PARALLEL_MAT_WRITE_VALUES {
        payload
            .par_chunks_mut(std::mem::size_of::<i32>())
            .enumerate()
            .for_each(|(dst, chunk)| {
                let col = dst / rows;
                let row = dst % rows;
                chunk.copy_from_slice(&values[row * cols + col].to_le_bytes());
            });
    } else {
        for (dst, chunk) in payload.chunks_mut(std::mem::size_of::<i32>()).enumerate() {
            let col = dst / rows;
            let row = dst % rows;
            chunk.copy_from_slice(&values[row * cols + col].to_le_bytes());
        }
    }
    pad_to_8(out, byte_len)
}

fn write_numeric_data_u32(
    out: &mut Vec<u8>,
    data_type: u32,
    rows: usize,
    cols: usize,
    values: &[u32],
) -> io::Result<()> {
    let byte_len = std::mem::size_of_val(values);
    write_tag(out, data_type, byte_len)?;
    let start = out.len();
    out.resize(start + byte_len, 0);
    let payload = &mut out[start..start + byte_len];
    if values.len() >= PARALLEL_MAT_WRITE_VALUES {
        payload
            .par_chunks_mut(std::mem::size_of::<u32>())
            .enumerate()
            .for_each(|(dst, chunk)| {
                let col = dst / rows;
                let row = dst % rows;
                chunk.copy_from_slice(&values[row * cols + col].to_le_bytes());
            });
    } else {
        for (dst, chunk) in payload.chunks_mut(std::mem::size_of::<u32>()).enumerate() {
            let col = dst / rows;
            let row = dst % rows;
            chunk.copy_from_slice(&values[row * cols + col].to_le_bytes());
        }
    }
    pad_to_8(out, byte_len)
}

fn write_numeric_data_u8(
    out: &mut Vec<u8>,
    data_type: u32,
    rows: usize,
    cols: usize,
    values: &[u8],
) -> io::Result<()> {
    let byte_len = values.len();
    write_tag(out, data_type, byte_len)?;
    out.reserve(byte_len + 8);
    for col in 0..cols {
        for row in 0..rows {
            out.push(values[row * cols + col]);
        }
    }
    pad_to_8(out, byte_len)
}

fn write_sparse_i32_data(out: &mut Vec<u8>, values: &[i32]) -> io::Result<()> {
    let byte_len = std::mem::size_of_val(values);
    write_tag(out, MI_INT32, byte_len)?;
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    pad_to_8(out, byte_len)
}

fn write_sparse_f64_data(out: &mut Vec<u8>, values: &[f64]) -> io::Result<()> {
    let byte_len = std::mem::size_of_val(values);
    write_tag(out, MI_DOUBLE, byte_len)?;
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    pad_to_8(out, byte_len)
}

fn write_complex_f32(matrix: &ComplexMatrixF32, out: &mut Vec<u8>) -> io::Result<()> {
    write_complex_f32_component(out, matrix.rows, matrix.cols, &matrix.values, true)?;
    write_complex_f32_component(out, matrix.rows, matrix.cols, &matrix.values, false)
}

fn write_complex_f32_component(
    out: &mut Vec<u8>,
    rows: usize,
    cols: usize,
    values: &[(f32, f32)],
    real: bool,
) -> io::Result<()> {
    let byte_len = values.len() * std::mem::size_of::<f32>();
    write_tag(out, MI_SINGLE, byte_len)?;
    let start = out.len();
    out.resize(start + byte_len, 0);
    let payload = &mut out[start..start + byte_len];
    if values.len() >= PARALLEL_MAT_WRITE_VALUES {
        payload
            .par_chunks_mut(std::mem::size_of::<f32>())
            .enumerate()
            .for_each(|(dst, chunk)| {
                let col = dst / rows;
                let row = dst % rows;
                let value = values[row * cols + col];
                let component = if real { value.0 } else { value.1 };
                chunk.copy_from_slice(&component.to_le_bytes());
            });
    } else {
        for (dst, chunk) in payload.chunks_mut(std::mem::size_of::<f32>()).enumerate() {
            let col = dst / rows;
            let row = dst % rows;
            let value = values[row * cols + col];
            let component = if real { value.0 } else { value.1 };
            chunk.copy_from_slice(&component.to_le_bytes());
        }
    }
    pad_to_8(out, byte_len)
}

fn write_complex_f32_array(array: &ComplexArrayF32, out: &mut Vec<u8>) -> io::Result<()> {
    let real: Vec<f32> = array.values.iter().map(|value| value.0).collect();
    let imag: Vec<f32> = array.values.iter().map(|value| value.1).collect();
    write_numeric_data_nd(out, MI_SINGLE, &array.dims, &real, write_f32_value)?;
    write_numeric_data_nd(out, MI_SINGLE, &array.dims, &imag, write_f32_value)
}

fn write_complex_f64(matrix: &ComplexMatrixF64, out: &mut Vec<u8>) -> io::Result<()> {
    write_complex_f64_component(out, matrix.rows, matrix.cols, &matrix.values, true)?;
    write_complex_f64_component(out, matrix.rows, matrix.cols, &matrix.values, false)
}

fn write_complex_f64_component(
    out: &mut Vec<u8>,
    rows: usize,
    cols: usize,
    values: &[(f64, f64)],
    real: bool,
) -> io::Result<()> {
    let byte_len = values.len() * std::mem::size_of::<f64>();
    write_tag(out, MI_DOUBLE, byte_len)?;
    let start = out.len();
    out.resize(start + byte_len, 0);
    let payload = &mut out[start..start + byte_len];
    if values.len() >= PARALLEL_MAT_WRITE_VALUES {
        payload
            .par_chunks_mut(std::mem::size_of::<f64>())
            .enumerate()
            .for_each(|(dst, chunk)| {
                let col = dst / rows;
                let row = dst % rows;
                let value = values[row * cols + col];
                let component = if real { value.0 } else { value.1 };
                chunk.copy_from_slice(&component.to_le_bytes());
            });
    } else {
        for (dst, chunk) in payload.chunks_mut(std::mem::size_of::<f64>()).enumerate() {
            let col = dst / rows;
            let row = dst % rows;
            let value = values[row * cols + col];
            let component = if real { value.0 } else { value.1 };
            chunk.copy_from_slice(&component.to_le_bytes());
        }
    }
    pad_to_8(out, byte_len)
}

fn write_numeric_data_nd<T>(
    out: &mut Vec<u8>,
    data_type: u32,
    dims: &[usize],
    values: &[T],
    write_value: fn(&mut Vec<u8>, &T) -> io::Result<()>,
) -> io::Result<()> {
    let byte_len = std::mem::size_of_val(values);
    write_tag(out, data_type, byte_len)?;
    let mut indices = vec![0usize; dims.len()];
    for _ in 0..values.len() {
        let source_ix = row_major_index(&indices, dims);
        write_value(out, &values[source_ix])?;
        increment_col_major_index(&mut indices, dims);
    }
    pad_to_8(out, byte_len)
}

fn row_major_index(indices: &[usize], dims: &[usize]) -> usize {
    let mut out = 0usize;
    for (&index, &dim) in indices.iter().zip(dims.iter()) {
        out = out * dim + index;
    }
    out
}

fn increment_col_major_index(indices: &mut [usize], dims: &[usize]) {
    for (index, &dim) in indices.iter_mut().zip(dims.iter()) {
        *index += 1;
        if *index < dim {
            return;
        }
        *index = 0;
    }
}

fn write_f32_value(out: &mut Vec<u8>, value: &f32) -> io::Result<()> {
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_tag<W: Write>(out: &mut W, data_type: u32, bytes: usize) -> io::Result<()> {
    out.write_all(&data_type.to_le_bytes())?;
    out.write_all(&(bytes as u32).to_le_bytes())
}

fn pad_to_8<W: Write>(out: &mut W, len: usize) -> io::Result<()> {
    let pad = (8 - (len % 8)) % 8;
    if pad > 0 {
        out.write_all(&vec![0; pad])?;
    }
    Ok(())
}

fn parse_mat_file(
    path: &Path,
    bytes: &[u8],
    selected_variables: Option<&[&str]>,
) -> Result<MatData, MatError> {
    if bytes.len() < 128 {
        return Err(MatError::InvalidHeader {
            path: path.to_path_buf(),
            message: format!("file has {} bytes, expected at least 128", bytes.len()),
        });
    }
    let marker = [bytes[126], bytes[127]];
    let endian = match &marker {
        b"IM" => Endian::Little,
        b"MI" => Endian::Big,
        _ => {
            return Err(MatError::UnsupportedEndian {
                path: path.to_path_buf(),
                marker,
            })
        }
    };
    let mut variables = BTreeMap::new();
    parse_top_level_elements(
        path,
        &bytes[128..],
        endian,
        selected_variables,
        &mut variables,
    )?;
    Ok(MatData { variables })
}

fn parse_top_level_elements(
    path: &Path,
    bytes: &[u8],
    endian: Endian,
    selected_variables: Option<&[&str]>,
    variables: &mut BTreeMap<String, MatArray>,
) -> Result<(), MatError> {
    let mut offset = 0;
    while offset < bytes.len() {
        let element = read_element(bytes, &mut offset, endian).map_err(|message| {
            MatError::MalformedFile {
                path: path.to_path_buf(),
                message,
            }
        })?;
        if element.data_type == 0 && element.data.is_empty() {
            break;
        }
        match element.data_type {
            MI_MATRIX => {
                if let Some(array) = parse_matrix_element(element.data, endian, selected_variables)?
                {
                    variables.insert(array.name.clone(), array);
                }
            }
            MI_COMPRESSED => {
                let decompressed = decompress_mat_element(path, element.data)?;
                parse_top_level_elements(
                    path,
                    &decompressed,
                    endian,
                    selected_variables,
                    variables,
                )?;
            }
            other => {
                return Err(MatError::UnsupportedDataType {
                    name: "<top-level>".to_string(),
                    data_type: other,
                });
            }
        }
    }
    Ok(())
}

fn decompress_mat_element(path: &Path, bytes: &[u8]) -> Result<Vec<u8>, MatError> {
    let mut decoder = ZlibDecoder::new(bytes);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|source| MatError::MalformedFile {
            path: path.to_path_buf(),
            message: format!("unable to decompress miCOMPRESSED element: {source}"),
        })?;
    Ok(decompressed)
}

fn find_hdf5_signature_offset(bytes: &[u8]) -> Option<usize> {
    let scan_len = bytes.len().min(HDF5_SIGNATURE_SCAN_BYTES);
    bytes[..scan_len]
        .windows(HDF5_SIGNATURE.len())
        .position(|window| window == HDF5_SIGNATURE)
}

fn read_hdf5_mat_file(
    path: &Path,
    offset: usize,
    variables: Option<&[&str]>,
) -> Result<MatData, MatError> {
    if let Ok(data) = read_hdf5_mat_payload(path, path, variables) {
        return Ok(data);
    }

    let temp_path = hdf5_temp_dir().join(format!(
        "pystamps-mat-hdf5-{}-{}.h5",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| MatError::MalformedFile {
                path: path.to_path_buf(),
                message: err.to_string(),
            })?
            .as_nanos()
    ));
    {
        let mut input = File::open(path).map_err(|source| MatError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        input
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|source| MatError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .map_err(|source| MatError::Write {
                path: temp_path.clone(),
                source,
            })?;
        let mut buffer = vec![0_u8; HDF5_TEMP_COPY_BUFFER_BYTES];
        loop {
            let read = input.read(&mut buffer).map_err(|source| MatError::Read {
                path: path.to_path_buf(),
                source,
            })?;
            if read == 0 {
                break;
            }
            output
                .write_all(&buffer[..read])
                .map_err(|source| MatError::Write {
                    path: temp_path.clone(),
                    source,
                })?;
        }
        output.flush().map_err(|source| MatError::Write {
            path: temp_path.clone(),
            source,
        })?;
    }
    let result = read_hdf5_mat_payload(path, &temp_path, variables);
    let _ = fs::remove_file(&temp_path);
    result
}

fn hdf5_temp_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("PYSTAMPS_HDF5_TMPDIR") {
        return PathBuf::from(path);
    }
    let shm = Path::new("/dev/shm");
    if shm.is_dir() {
        return shm.to_path_buf();
    }
    std::env::temp_dir()
}

fn read_hdf5_mat_payload(
    source_path: &Path,
    hdf5_path: &Path,
    variables: Option<&[&str]>,
) -> Result<MatData, MatError> {
    let file = rust_hdf5::H5File::open(hdf5_path).map_err(|err| MatError::MalformedFile {
        path: source_path.to_path_buf(),
        message: format!("unable to open HDF5 MAT payload: {err}"),
    })?;
    let names = variables
        .map(|names| names.iter().map(|name| (*name).to_string()).collect())
        .unwrap_or_else(|| file.dataset_names());
    let mut variables = BTreeMap::new();
    for name in names {
        let Ok(dataset) = file.dataset(&name) else {
            continue;
        };
        let array = read_hdf5_dataset(source_path, &dataset, &name)?;
        variables.insert(name, array);
    }
    Ok(MatData { variables })
}

fn read_hdf5_dataset(
    source_path: &Path,
    dataset: &rust_hdf5::H5Dataset,
    name: &str,
) -> Result<MatArray, MatError> {
    if let Some(matlab_class) = hdf5_matlab_class(dataset) {
        match (matlab_class.as_str(), dataset.element_size()) {
            ("single", 8) => return read_hdf5_complex_f32_dataset(dataset, name),
            ("double", 16) => return read_hdf5_complex_f64_dataset(dataset, name),
            ("single", 4) => {
                if let Ok(values) = dataset.read_raw::<f32>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::F32(values)));
                }
            }
            ("double", 8) => {
                if let Ok(values) = dataset.read_raw::<f64>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::F64(values)));
                }
            }
            ("uint8", 1) => {
                if let Ok(values) = dataset.read_raw::<u8>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::U8(values)));
                }
            }
            ("int8", 1) => {
                if let Ok(values) = dataset.read_raw::<i8>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::I8(values)));
                }
            }
            ("uint16", 2) => {
                if let Ok(values) = dataset.read_raw::<u16>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::U16(values)));
                }
            }
            ("int16", 2) => {
                if let Ok(values) = dataset.read_raw::<i16>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::I16(values)));
                }
            }
            ("uint32", 4) => {
                if let Ok(values) = dataset.read_raw::<u32>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::U32(values)));
                }
            }
            ("int32", 4) => {
                if let Ok(values) = dataset.read_raw::<i32>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::I32(values)));
                }
            }
            ("uint64", 8) => {
                if let Ok(values) = dataset.read_raw::<u64>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::U64(values)));
                }
            }
            ("int64", 8) => {
                if let Ok(values) = dataset.read_raw::<i64>() {
                    return Ok(real_hdf5_array(name, dataset, NumericData::I64(values)));
                }
            }
            _ => {}
        }
    }

    if let Ok(values) = dataset.read_raw::<f64>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::F64(values)));
    }
    if let Ok(values) = dataset.read_raw::<f32>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::F32(values)));
    }
    if let Ok(values) = dataset.read_raw::<u8>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::U8(values)));
    }
    if let Ok(values) = dataset.read_raw::<i8>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::I8(values)));
    }
    if let Ok(values) = dataset.read_raw::<u16>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::U16(values)));
    }
    if let Ok(values) = dataset.read_raw::<i16>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::I16(values)));
    }
    if let Ok(values) = dataset.read_raw::<u32>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::U32(values)));
    }
    if let Ok(values) = dataset.read_raw::<i32>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::I32(values)));
    }
    if let Ok(values) = dataset.read_raw::<u64>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::U64(values)));
    }
    if let Ok(values) = dataset.read_raw::<i64>() {
        return Ok(real_hdf5_array(name, dataset, NumericData::I64(values)));
    }
    if let Ok(array) = read_hdf5_complex_f64_dataset(dataset, name) {
        return Ok(array);
    }
    if let Ok(array) = read_hdf5_complex_f32_dataset(dataset, name) {
        return Ok(array);
    }
    Err(MatError::UnsupportedDataType {
        name: format!("HDF5 dataset {name} in {}", source_path.display()),
        data_type: 0,
    })
}

fn hdf5_matlab_class(dataset: &rust_hdf5::H5Dataset) -> Option<String> {
    let attr = dataset.attr("MATLAB_class").ok()?;
    attr.read_string()
        .ok()
        .map(|class| class.trim_matches(char::from(0)).trim().to_string())
}

fn read_hdf5_complex_f32_dataset(
    dataset: &rust_hdf5::H5Dataset,
    name: &str,
) -> Result<MatArray, MatError> {
    let values =
        dataset
            .read_raw::<rust_hdf5::Complex32>()
            .map_err(|_| MatError::UnsupportedDataType {
                name: name.to_string(),
                data_type: 0,
            })?;
    let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
    let (real, imag): (Vec<f32>, Vec<f32>) =
        values.into_iter().map(|value| (value.re, value.im)).unzip();
    Ok(MatArray {
        name: name.to_string(),
        rows,
        cols,
        numeric_type: NumericType::F32,
        real: NumericData::F32(real),
        imag: Some(NumericData::F32(imag)),
    })
}

fn read_hdf5_complex_f64_dataset(
    dataset: &rust_hdf5::H5Dataset,
    name: &str,
) -> Result<MatArray, MatError> {
    let values =
        dataset
            .read_raw::<rust_hdf5::Complex64>()
            .map_err(|_| MatError::UnsupportedDataType {
                name: name.to_string(),
                data_type: 0,
            })?;
    let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
    let (real, imag): (Vec<f64>, Vec<f64>) =
        values.into_iter().map(|value| (value.re, value.im)).unzip();
    Ok(MatArray {
        name: name.to_string(),
        rows,
        cols,
        numeric_type: NumericType::F64,
        real: NumericData::F64(real),
        imag: Some(NumericData::F64(imag)),
    })
}

fn real_hdf5_array(name: &str, dataset: &rust_hdf5::H5Dataset, data: NumericData) -> MatArray {
    match data {
        NumericData::F64(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::F64,
                real: NumericData::F64(values),
                imag: None,
            }
        }
        NumericData::F32(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::F32,
                real: NumericData::F32(values),
                imag: None,
            }
        }
        NumericData::I8(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::I8,
                real: NumericData::I8(values),
                imag: None,
            }
        }
        NumericData::U8(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::U8,
                real: NumericData::U8(values),
                imag: None,
            }
        }
        NumericData::I16(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::I16,
                real: NumericData::I16(values),
                imag: None,
            }
        }
        NumericData::U16(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::U16,
                real: NumericData::U16(values),
                imag: None,
            }
        }
        NumericData::I32(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::I32,
                real: NumericData::I32(values),
                imag: None,
            }
        }
        NumericData::U32(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::U32,
                real: NumericData::U32(values),
                imag: None,
            }
        }
        NumericData::I64(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::I64,
                real: NumericData::I64(values),
                imag: None,
            }
        }
        NumericData::U64(values) => {
            let (rows, cols, values) = orient_hdf5_dataset_values(dataset, values);
            MatArray {
                name: name.to_string(),
                rows,
                cols,
                numeric_type: NumericType::U64,
                real: NumericData::U64(values),
                imag: None,
            }
        }
    }
}

fn orient_hdf5_dataset_values<T: Copy>(
    dataset: &rust_hdf5::H5Dataset,
    values: Vec<T>,
) -> (usize, usize, Vec<T>) {
    let shape = dataset.shape();
    if hdf5_pystamps_row_major(dataset) {
        return orient_row_major_hdf5_values(&shape, values);
    }
    orient_hdf5_values(&shape, values)
}

fn orient_row_major_hdf5_values<T: Copy>(
    shape: &[usize],
    values: Vec<T>,
) -> (usize, usize, Vec<T>) {
    match shape {
        [] => (1, 1, values),
        [_] => (values.len(), 1, values),
        [rows, cols] => (*rows, *cols, values),
        _ => {
            let rows = *shape.first().unwrap_or(&1);
            let cols = values.len() / rows.max(1);
            (rows, cols, values)
        }
    }
}

fn hdf5_pystamps_row_major(dataset: &rust_hdf5::H5Dataset) -> bool {
    dataset
        .attr("PY_STAMPS_row_major")
        .ok()
        .and_then(|attr| attr.read_numeric::<u8>().ok())
        .is_some_and(|value| value != 0)
}

fn orient_hdf5_values<T: Copy>(shape: &[usize], values: Vec<T>) -> (usize, usize, Vec<T>) {
    match shape {
        [] => (1, 1, values),
        [_] => (values.len(), 1, values),
        [raw_rows, raw_cols] => {
            let mut out = Vec::with_capacity(values.len());
            for row in 0..*raw_cols {
                for col in 0..*raw_rows {
                    out.push(values[col * *raw_cols + row]);
                }
            }
            (*raw_cols, *raw_rows, out)
        }
        _ => {
            let rows = *shape.last().unwrap_or(&1);
            let cols = values.len() / rows.max(1);
            (rows, cols, values)
        }
    }
}

fn parse_matrix_element(
    bytes: &[u8],
    endian: Endian,
    selected_variables: Option<&[&str]>,
) -> Result<Option<MatArray>, MatError> {
    let mut offset = 0;
    let flags = read_element(bytes, &mut offset, endian).map_err(|message| {
        MatError::MalformedVariable {
            name: "<unknown>".to_string(),
            message,
        }
    })?;
    if flags.data_type != MI_UINT32 || flags.data.len() < 4 {
        return Err(MatError::MalformedVariable {
            name: "<unknown>".to_string(),
            message: "missing array flags".to_string(),
        });
    }
    let flag_word = endian.read_u32(&flags.data[..4]);
    let class = flag_word & 0xff;
    let is_complex = (flag_word & MX_COMPLEX_FLAG) != 0;
    let _is_logical = (flag_word & MX_LOGICAL_FLAG) != 0;

    let dims_element = read_element(bytes, &mut offset, endian).map_err(|message| {
        MatError::MalformedVariable {
            name: "<unknown>".to_string(),
            message,
        }
    })?;
    if dims_element.data_type != MI_INT32 || dims_element.data.len() % 4 != 0 {
        return Err(MatError::MalformedVariable {
            name: "<unknown>".to_string(),
            message: "missing int32 dimensions".to_string(),
        });
    }
    let dims = dims_element
        .data
        .chunks_exact(4)
        .map(|chunk| endian.read_i32(chunk))
        .collect::<Vec<_>>();

    let name_element = read_element(bytes, &mut offset, endian).map_err(|message| {
        MatError::MalformedVariable {
            name: "<unknown>".to_string(),
            message,
        }
    })?;
    let name = if name_element.data_type == MI_INT8 || name_element.data_type == MI_UINT8 {
        String::from_utf8_lossy(name_element.data).to_string()
    } else {
        "<unknown>".to_string()
    };
    if selected_variables.is_some_and(|selected| !selected.contains(&name.as_str())) {
        return Ok(None);
    }

    if dims.len() < 2 || dims.iter().any(|&dim| dim < 0) {
        return Err(MatError::MalformedDimensions { name, dims });
    }
    let rows = dims[0] as usize;
    let mut cols = 1_usize;
    for dim in &dims[1..] {
        cols = cols
            .checked_mul(*dim as usize)
            .ok_or_else(|| MatError::MalformedVariable {
                name: name.clone(),
                message: "dimensions overflow usize".to_string(),
            })?;
    }
    let Some(expected_len) = rows.checked_mul(cols) else {
        return Err(MatError::MalformedVariable {
            name,
            message: "dimensions overflow usize".to_string(),
        });
    };
    if class == MX_SPARSE_CLASS || !supported_class(class) {
        return Err(MatError::UnsupportedClass { name, class });
    }

    let real_element = read_element(bytes, &mut offset, endian).map_err(|message| {
        MatError::MalformedVariable {
            name: name.clone(),
            message,
        }
    })?;
    let real = decode_numeric_data(
        &name,
        real_element.data_type,
        real_element.data,
        endian,
        rows,
        cols,
        expected_len,
    )?;
    let numeric_type = real.numeric_type();
    let imag = if is_complex {
        let imag_element = read_element(bytes, &mut offset, endian).map_err(|message| {
            MatError::MalformedVariable {
                name: name.clone(),
                message,
            }
        })?;
        let imag = decode_numeric_data(
            &name,
            imag_element.data_type,
            imag_element.data,
            endian,
            rows,
            cols,
            expected_len,
        )?;
        if imag.numeric_type() != numeric_type {
            return Err(MatError::MalformedVariable {
                name,
                message: "real and imaginary payload types differ".to_string(),
            });
        }
        Some(imag)
    } else {
        None
    };

    Ok(Some(MatArray {
        name,
        rows,
        cols,
        numeric_type,
        real,
        imag,
    }))
}

fn decode_numeric_data(
    name: &str,
    data_type: u32,
    bytes: &[u8],
    endian: Endian,
    rows: usize,
    cols: usize,
    expected_len: usize,
) -> Result<NumericData, MatError> {
    match data_type {
        MI_DOUBLE => decode_fixed_width(name, bytes, expected_len, 8, |chunk| {
            NumericValue::F64(endian.read_f64(chunk))
        })
        .map(|values| NumericData::F64(row_major_f64(values, rows, cols))),
        MI_SINGLE => decode_fixed_width(name, bytes, expected_len, 4, |chunk| {
            NumericValue::F32(endian.read_f32(chunk))
        })
        .map(|values| NumericData::F32(row_major_f32(values, rows, cols))),
        MI_INT8 => decode_i8(name, bytes, expected_len)
            .map(|values| NumericData::I8(row_major_i8(values, rows, cols))),
        MI_UINT8 => decode_u8(name, bytes, expected_len)
            .map(|values| NumericData::U8(row_major_u8(values, rows, cols))),
        MI_INT16 => decode_fixed_width(name, bytes, expected_len, 2, |chunk| {
            NumericValue::I16(endian.read_i16(chunk))
        })
        .map(|values| NumericData::I16(row_major_i16(values, rows, cols))),
        MI_UINT16 => decode_fixed_width(name, bytes, expected_len, 2, |chunk| {
            NumericValue::U16(endian.read_u16(chunk))
        })
        .map(|values| NumericData::U16(row_major_u16(values, rows, cols))),
        MI_INT32 => decode_fixed_width(name, bytes, expected_len, 4, |chunk| {
            NumericValue::I32(endian.read_i32(chunk))
        })
        .map(|values| NumericData::I32(row_major_i32(values, rows, cols))),
        MI_UINT32 => decode_fixed_width(name, bytes, expected_len, 4, |chunk| {
            NumericValue::U32(endian.read_u32(chunk))
        })
        .map(|values| NumericData::U32(row_major_u32(values, rows, cols))),
        MI_INT64 => decode_fixed_width(name, bytes, expected_len, 8, |chunk| {
            NumericValue::I64(endian.read_i64(chunk))
        })
        .map(|values| NumericData::I64(row_major_i64(values, rows, cols))),
        MI_UINT64 => decode_fixed_width(name, bytes, expected_len, 8, |chunk| {
            NumericValue::U64(endian.read_u64(chunk))
        })
        .map(|values| NumericData::U64(row_major_u64(values, rows, cols))),
        MI_UTF8 => decode_u8(name, bytes, expected_len)
            .map(|values| NumericData::U8(row_major_u8(values, rows, cols))),
        MI_UTF16 => decode_fixed_width(name, bytes, expected_len, 2, |chunk| {
            NumericValue::U16(endian.read_u16(chunk))
        })
        .map(|values| NumericData::U16(row_major_u16(values, rows, cols))),
        MI_UTF32 => decode_fixed_width(name, bytes, expected_len, 4, |chunk| {
            NumericValue::U32(endian.read_u32(chunk))
        })
        .map(|values| NumericData::U32(row_major_u32(values, rows, cols))),
        other => Err(MatError::UnsupportedDataType {
            name: name.to_string(),
            data_type: other,
        }),
    }
}

#[derive(Clone, Copy)]
enum NumericValue {
    F64(f64),
    F32(f32),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
}

fn decode_fixed_width(
    name: &str,
    bytes: &[u8],
    expected_len: usize,
    width: usize,
    decode: impl Fn(&[u8]) -> NumericValue,
) -> Result<Vec<NumericValue>, MatError> {
    let expected_bytes =
        expected_len
            .checked_mul(width)
            .ok_or_else(|| MatError::MalformedVariable {
                name: name.to_string(),
                message: "payload byte count overflow".to_string(),
            })?;
    if bytes.len() != expected_bytes {
        return Err(MatError::MalformedVariable {
            name: name.to_string(),
            message: format!(
                "payload has {} bytes, expected {expected_bytes}",
                bytes.len()
            ),
        });
    }
    Ok(bytes.chunks_exact(width).map(decode).collect())
}

fn decode_i8(name: &str, bytes: &[u8], expected_len: usize) -> Result<Vec<i8>, MatError> {
    if bytes.len() != expected_len {
        return Err(MatError::MalformedVariable {
            name: name.to_string(),
            message: format!("payload has {} bytes, expected {expected_len}", bytes.len()),
        });
    }
    Ok(bytes.iter().map(|&value| value as i8).collect())
}

fn decode_u8(name: &str, bytes: &[u8], expected_len: usize) -> Result<Vec<u8>, MatError> {
    if bytes.len() != expected_len {
        return Err(MatError::MalformedVariable {
            name: name.to_string(),
            message: format!("payload has {} bytes, expected {expected_len}", bytes.len()),
        });
    }
    Ok(bytes.to_vec())
}

macro_rules! row_major_from_numeric_value {
    ($func:ident, $variant:ident, $ty:ty) => {
        fn $func(values: Vec<NumericValue>, rows: usize, cols: usize) -> Vec<$ty> {
            let mut out = vec![0 as $ty; values.len()];
            for col in 0..cols {
                for row in 0..rows {
                    let NumericValue::$variant(value) = values[col * rows + row] else {
                        unreachable!("decoder produced the requested numeric variant")
                    };
                    out[row * cols + col] = value;
                }
            }
            out
        }
    };
}

row_major_from_numeric_value!(row_major_f64, F64, f64);
row_major_from_numeric_value!(row_major_f32, F32, f32);
row_major_from_numeric_value!(row_major_i16, I16, i16);
row_major_from_numeric_value!(row_major_u16, U16, u16);
row_major_from_numeric_value!(row_major_i32, I32, i32);
row_major_from_numeric_value!(row_major_u32, U32, u32);
row_major_from_numeric_value!(row_major_i64, I64, i64);
row_major_from_numeric_value!(row_major_u64, U64, u64);

fn row_major_i8(values: Vec<i8>, rows: usize, cols: usize) -> Vec<i8> {
    row_major_copy(values, rows, cols)
}

fn row_major_u8(values: Vec<u8>, rows: usize, cols: usize) -> Vec<u8> {
    row_major_copy(values, rows, cols)
}

fn row_major_copy<T: Copy + Default>(values: Vec<T>, rows: usize, cols: usize) -> Vec<T> {
    let mut out = vec![T::default(); values.len()];
    for col in 0..cols {
        for row in 0..rows {
            out[row * cols + col] = values[col * rows + row];
        }
    }
    out
}

fn supported_class(class: u32) -> bool {
    matches!(
        class,
        MX_DOUBLE_CLASS
            | MX_CHAR_CLASS
            | MX_SINGLE_CLASS
            | MX_INT8_CLASS
            | MX_UINT8_CLASS
            | MX_INT16_CLASS
            | MX_UINT16_CLASS
            | MX_INT32_CLASS
            | MX_UINT32_CLASS
            | MX_INT64_CLASS
            | MX_UINT64_CLASS
    )
}

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn read_u16(self, bytes: &[u8]) -> u16 {
        let chunk: [u8; 2] = bytes[..2].try_into().expect("caller checked width");
        match self {
            Endian::Little => u16::from_le_bytes(chunk),
            Endian::Big => u16::from_be_bytes(chunk),
        }
    }

    fn read_i16(self, bytes: &[u8]) -> i16 {
        self.read_u16(bytes) as i16
    }

    fn read_u32(self, bytes: &[u8]) -> u32 {
        let chunk: [u8; 4] = bytes[..4].try_into().expect("caller checked width");
        match self {
            Endian::Little => u32::from_le_bytes(chunk),
            Endian::Big => u32::from_be_bytes(chunk),
        }
    }

    fn read_i32(self, bytes: &[u8]) -> i32 {
        self.read_u32(bytes) as i32
    }

    fn read_u64(self, bytes: &[u8]) -> u64 {
        let chunk: [u8; 8] = bytes[..8].try_into().expect("caller checked width");
        match self {
            Endian::Little => u64::from_le_bytes(chunk),
            Endian::Big => u64::from_be_bytes(chunk),
        }
    }

    fn read_i64(self, bytes: &[u8]) -> i64 {
        self.read_u64(bytes) as i64
    }

    fn read_f32(self, bytes: &[u8]) -> f32 {
        f32::from_bits(self.read_u32(bytes))
    }

    fn read_f64(self, bytes: &[u8]) -> f64 {
        f64::from_bits(self.read_u64(bytes))
    }
}

struct DataElement<'a> {
    data_type: u32,
    data: &'a [u8],
}

fn read_element<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    endian: Endian,
) -> Result<DataElement<'a>, String> {
    if *offset == bytes.len() {
        return Ok(DataElement {
            data_type: 0,
            data: &[],
        });
    }
    if bytes.len() - *offset < 8 {
        return Err(format!("truncated tag at byte {}", *offset));
    }

    let small_type = endian.read_u16(&bytes[*offset..*offset + 2]) as u32;
    let small_size = endian.read_u16(&bytes[*offset + 2..*offset + 4]) as u32;
    if small_size > 0 {
        if small_size > 4 {
            return Err(format!(
                "small data element at byte {} has {small_size} bytes",
                *offset
            ));
        }
        let data_start = *offset + 4;
        let data_end = data_start + small_size as usize;
        if data_end > bytes.len() {
            return Err(format!(
                "small data element at byte {} exceeds file length",
                *offset
            ));
        }
        *offset += 8;
        return Ok(DataElement {
            data_type: small_type,
            data: &bytes[data_start..data_end],
        });
    }

    let data_type = endian.read_u32(&bytes[*offset..*offset + 4]);
    let data_size = endian.read_u32(&bytes[*offset + 4..*offset + 8]) as usize;
    let data_start = *offset + 8;
    let data_end = data_start
        .checked_add(data_size)
        .ok_or_else(|| format!("data element at byte {} overflows usize", *offset))?;
    if data_end > bytes.len() {
        return Err(format!(
            "data element at byte {} has {} bytes but only {} remain",
            *offset,
            data_size,
            bytes.len().saturating_sub(data_start)
        ));
    }
    let padded_end = data_end
        .checked_add((8 - (data_size % 8)) % 8)
        .ok_or_else(|| {
            format!(
                "data element at byte {} padded length overflows usize",
                *offset
            )
        })?;
    if padded_end > bytes.len() {
        return Err(format!(
            "data element at byte {} padding exceeds file length",
            *offset
        ));
    }
    *offset = padded_end;
    Ok(DataElement {
        data_type,
        data: &bytes[data_start..data_end],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::process::Command;

    #[test]
    fn rejects_shape_mismatch() {
        let mut mat = MatFile::new("unused.mat");
        let err = mat
            .add_f64_matrix("x", 2, 2, vec![1.0, 2.0, 3.0])
            .unwrap_err();
        assert!(err.to_string().contains("3 values for 2x2"));
    }

    #[test]
    fn writes_and_reads_mat_v5_numeric_shapes() {
        let path = temp_path("pystamps-mat-roundtrip");
        let mut mat = MatFile::new(&path);
        mat.add_f64_scalar("scalar", 7.5).unwrap();
        mat.add_f32_row_vector("row", vec![1.0, 2.0, 3.0]).unwrap();
        mat.add_i32_matrix("col", 3, 1, vec![4, 5, 6]).unwrap();
        mat.add_f64_matrix("matrix", 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        mat.write().unwrap();

        let data = MatData::read(&path).unwrap();
        let scalar = data.get_f64_matrix("scalar").unwrap();
        assert_eq!((scalar.rows, scalar.cols, scalar.values), (1, 1, vec![7.5]));
        let row = data.get_f32_matrix("row").unwrap();
        assert_eq!(
            (row.rows, row.cols, row.values),
            (1, 3, vec![1.0, 2.0, 3.0])
        );
        let col = data.get_f64_matrix("col").unwrap();
        assert_eq!(
            (col.rows, col.cols, col.values),
            (3, 1, vec![4.0, 5.0, 6.0])
        );
        let matrix = data.get_f64_matrix("matrix").unwrap();
        assert_eq!((matrix.rows, matrix.cols), (2, 3));
        assert_eq!(matrix.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_compressed_mat_v5_elements() {
        let path = temp_path("pystamps-mat-compressed");
        let mut mat = MatFile::new(&path);
        mat.add_f64_col_vector("n_ps", vec![3.0]).unwrap();
        mat.add_f64_matrix(
            "ij",
            3,
            3,
            vec![1.0, 10.0, 20.0, 2.0, 11.0, 21.0, 3.0, 12.0, 22.0],
        )
        .unwrap();
        mat.write().unwrap();

        let original = std::fs::read(&path).unwrap();
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original[128..]).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut bytes = original[..128].to_vec();
        write_tag(&mut bytes, MI_COMPRESSED, compressed.len()).unwrap();
        bytes.extend_from_slice(&compressed);
        pad_to_8(&mut bytes, compressed.len()).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let data = MatData::read(&path).unwrap();
        let n_ps = data.get_f64_matrix("n_ps").unwrap();
        assert_eq!(n_ps.values, vec![3.0]);
        let ij = data.get_f64_matrix("ij").unwrap();
        assert_eq!((ij.rows, ij.cols), (3, 3));
        assert_eq!(ij.values[8], 22.0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_matlab_hdf5_userblock_numeric_dataset() {
        let path = temp_path("pystamps-mat-hdf5-userblock");
        let raw_path = path.with_extension("h5");
        let h5 = rust_hdf5::H5File::create(&raw_path).unwrap();
        let ds = h5
            .new_dataset::<f64>()
            .shape(&[3usize, 2usize])
            .create("ij")
            .unwrap();
        ds.write_raw(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        h5.close().unwrap();

        let mut matlab_hdf5 = File::create(&path).unwrap();
        matlab_hdf5.write_all(&vec![b' '; 512]).unwrap();
        matlab_hdf5
            .write_all(&std::fs::read(&raw_path).unwrap())
            .unwrap();
        std::fs::remove_file(&raw_path).unwrap();

        let data = MatData::read(&path).unwrap();
        let ij = data.get_f64_matrix("ij").unwrap();
        assert_eq!((ij.rows, ij.cols), (2, 3));
        assert_eq!(ij.values, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_matlab_hdf5_userblock_complex_single_dataset() {
        let path = temp_path("pystamps-mat-hdf5-complex-userblock");
        let raw_path = path.with_extension("h5");
        let h5 = rust_hdf5::H5File::create(&raw_path).unwrap();
        let ds = h5
            .new_dataset::<rust_hdf5::Complex32>()
            .shape(&[2usize, 3usize])
            .create("ph_patch")
            .unwrap();
        ds.write_raw(&[
            rust_hdf5::Complex32 { re: 1.0, im: -1.0 },
            rust_hdf5::Complex32 { re: 2.0, im: -2.0 },
            rust_hdf5::Complex32 { re: 3.0, im: -3.0 },
            rust_hdf5::Complex32 { re: 4.0, im: -4.0 },
            rust_hdf5::Complex32 { re: 5.0, im: -5.0 },
            rust_hdf5::Complex32 { re: 6.0, im: -6.0 },
        ])
        .unwrap();
        ds.new_attr::<rust_hdf5::VarLenUnicode>()
            .shape(())
            .create("MATLAB_class")
            .unwrap()
            .write_string("single")
            .unwrap();
        h5.close().unwrap();

        let mut matlab_hdf5 = File::create(&path).unwrap();
        matlab_hdf5.write_all(&vec![b' '; 512]).unwrap();
        matlab_hdf5
            .write_all(&std::fs::read(&raw_path).unwrap())
            .unwrap();
        std::fs::remove_file(&raw_path).unwrap();

        let data = MatData::read(&path).unwrap();
        let ph_patch = data.get_complex_f32_matrix("ph_patch").unwrap();
        assert_eq!((ph_patch.rows, ph_patch.cols), (3, 2));
        assert_eq!(
            ph_patch.values,
            vec![
                (1.0, -1.0),
                (4.0, -4.0),
                (2.0, -2.0),
                (5.0, -5.0),
                (3.0, -3.0),
                (6.0, -6.0),
            ]
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn writes_complex_ph_matrix_readable_by_scipy() {
        let path = temp_path("pystamps-mat-scipy-ph");
        write_phase_artifact(
            &path,
            2,
            2,
            vec![(1.0, 2.0), (3.0, 4.0), (5.0, -6.0), (7.0, -8.0)],
        )
        .unwrap();

        let data = MatData::read(&path).unwrap();
        let ph = data.get_complex_f32_matrix(VAR_PH).unwrap();
        assert_eq!((ph.rows, ph.cols), (2, 2));
        assert_eq!(ph.values[2], (5.0, -6.0));

        let script = format!(
            "import numpy as np; from scipy.io import loadmat; ph=loadmat({path:?})['ph']; assert ph.shape == (2, 2); np.testing.assert_allclose(ph, np.array([[1+2j, 3+4j], [5-6j, 7-8j]], dtype=np.complex64))",
            path = path.to_string_lossy()
        );
        let status = Command::new("uv")
            .args(["run", "python", "-c", &script])
            .status()
            .expect("uv run python should be available for pySTAMPS tests");
        assert!(
            status.success(),
            "scipy.io.loadmat failed for Rust-written ph matrix"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn writes_sparse_csc_f64_readable_by_scipy() {
        let path = temp_path("pystamps-mat-scipy-sparse");
        let mut mat = MatFile::new(&path);
        mat.add_f64_scalar("dense", 7.0).unwrap();
        mat.add_empty_sparse_csc_f64_matrix("spread", 3, 4).unwrap();
        mat.write().unwrap();

        let selected = MatData::read_selected(&path, &["dense"]).unwrap();
        assert_eq!(selected.get_f64_matrix("dense").unwrap().values, vec![7.0]);
        assert!(matches!(
            selected.get("spread").unwrap_err(),
            MatError::MissingVariable { .. }
        ));

        let script = format!(
            "import numpy as np; from scipy import sparse; from scipy.io import loadmat; spread=loadmat({path:?})['spread']; csc=spread.tocsc(); assert sparse.issparse(spread); assert csc.shape == (3, 4); assert csc.nnz == 0; np.testing.assert_array_equal(csc.indptr, np.zeros(5, dtype=csc.indptr.dtype))",
            path = path.to_string_lossy()
        );
        let status = Command::new("uv")
            .args(["run", "python", "-c", &script])
            .status()
            .expect("uv run python should be available for pySTAMPS tests");
        assert!(
            status.success(),
            "scipy.io.loadmat failed for Rust-written sparse matrix"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn writes_complex_3d_array_readable_by_scipy() {
        let path = temp_path("pystamps-mat-scipy-3d");
        let mut mat = MatFile::new(&path);
        mat.add_complex_f32_array3(
            "ph_grid",
            2,
            3,
            2,
            vec![
                (1.0, 0.0),
                (2.0, 0.0),
                (3.0, 0.0),
                (4.0, 0.0),
                (5.0, 0.0),
                (6.0, 0.0),
                (7.0, 0.0),
                (8.0, 0.0),
                (9.0, 0.0),
                (10.0, 0.0),
                (11.0, 0.0),
                (12.0, 0.0),
            ],
        )
        .unwrap();
        mat.write().unwrap();

        let data = MatData::read(&path).unwrap();
        let flat = data.get_complex_f32_matrix("ph_grid").unwrap();
        assert_eq!((flat.rows, flat.cols), (2, 6));
        assert_eq!(flat.values[5], (6.0, 0.0));

        let script = format!(
            "import numpy as np; from scipy.io import loadmat; ph=loadmat({path:?})['ph_grid']; assert ph.shape == (2, 3, 2); np.testing.assert_allclose(ph.reshape(-1), np.arange(1, 13, dtype=np.complex64))",
            path = path.to_string_lossy()
        );
        let status = Command::new("uv")
            .args(["run", "python", "-c", &script])
            .status()
            .expect("uv run python should be available for pySTAMPS tests");
        assert!(
            status.success(),
            "scipy.io.loadmat failed for Rust-written 3-D complex array"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_matlab_char_arrays_as_numeric_codes() {
        let path = temp_path("pystamps-mat-char");
        let script = format!(
            "from scipy.io import savemat; savemat({path:?}, {{'small_baseline_flag': 'n'}})",
            path = path.to_string_lossy()
        );
        let status = Command::new("uv")
            .args(["run", "python", "-c", &script])
            .status()
            .expect("uv run python should be available for pySTAMPS tests");
        assert!(status.success());

        let data = MatData::read(&path).unwrap();
        let flag = data.get_f64_matrix("small_baseline_flag").unwrap();
        assert_eq!((flag.rows, flag.cols), (1, 1));
        assert_eq!(flag.values, vec!['n' as u32 as f64]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_three_dimensional_payloads_as_flattened_matrices() {
        let path = temp_path("pystamps-mat-3d");
        let mut bytes = Vec::new();
        write_header_to_vec(&mut bytes);

        let mut body = Vec::new();
        write_array_flags(&mut body, MX_DOUBLE_CLASS, false).unwrap();
        write_tag(&mut body, MI_INT32, 12).unwrap();
        body.write_all(&1i32.to_le_bytes()).unwrap();
        body.write_all(&2i32.to_le_bytes()).unwrap();
        body.write_all(&3i32.to_le_bytes()).unwrap();
        pad_to_8(&mut body, 12).unwrap();
        write_name(&mut body, "flat3").unwrap();
        write_tag(&mut body, MI_DOUBLE, 48).unwrap();
        for value in [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
            body.write_all(&value.to_le_bytes()).unwrap();
        }
        write_tag(&mut bytes, MI_MATRIX, body.len()).unwrap();
        bytes.extend_from_slice(&body);
        pad_to_8(&mut bytes, body.len()).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let data = MatData::read(&path).unwrap();
        let flat3 = data.get_f64_matrix("flat3").unwrap();
        assert_eq!((flat3.rows, flat3.cols), (1, 6));
        assert_eq!(flat3.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_dimensions_return_structured_error() {
        let path = temp_path("pystamps-mat-malformed-dims");
        let mut bytes = Vec::new();
        write_header_to_vec(&mut bytes);

        let mut body = Vec::new();
        write_array_flags(&mut body, MX_DOUBLE_CLASS, false).unwrap();
        write_tag(&mut body, MI_INT32, 4).unwrap();
        body.write_all(&6i32.to_le_bytes()).unwrap();
        pad_to_8(&mut body, 4).unwrap();
        write_name(&mut body, "bad").unwrap();
        write_tag(&mut body, MI_DOUBLE, 48).unwrap();
        for value in [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
            body.write_all(&value.to_le_bytes()).unwrap();
        }
        write_tag(&mut bytes, MI_MATRIX, body.len()).unwrap();
        bytes.extend_from_slice(&body);
        pad_to_8(&mut bytes, body.len()).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let err = MatData::read(&path).unwrap_err();
        assert!(matches!(err, MatError::MalformedDimensions { .. }));
        std::fs::remove_file(path).unwrap();
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("{name}-{}.mat", std::process::id()))
    }

    fn write_header_to_vec(out: &mut Vec<u8>) {
        let mut text = [b' '; 116];
        let description = b"MATLAB 5.0 MAT-file, Platform: pySTAMPS Rust native";
        text[..description.len()].copy_from_slice(description);
        out.extend_from_slice(&text);
        out.extend_from_slice(&[0; 8]);
        out.extend_from_slice(&0x0100u16.to_le_bytes());
        out.extend_from_slice(b"IM");
    }
}
