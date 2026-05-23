use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

const MI_INT8: u32 = 1;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_MATRIX: u32 = 14;

const MX_DOUBLE_CLASS: u32 = 6;
const MX_SINGLE_CLASS: u32 = 7;
const MX_COMPLEX_FLAG: u32 = 0x0800;

#[derive(Debug, Error)]
pub enum MatError {
    #[error("unable to write MAT file {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("MAT variable {name} has {actual} values for {rows}x{cols}")]
    Shape {
        name: String,
        rows: usize,
        cols: usize,
        actual: usize,
    },
}

pub struct MatFile {
    path: PathBuf,
    variables: Vec<MatVar>,
}

enum MatVar {
    F64(Matrix<f64>),
    F32(Matrix<f32>),
    ComplexF32(ComplexMatrixF32),
}

pub struct Matrix<T> {
    name: String,
    rows: usize,
    cols: usize,
    values: Vec<T>,
}

pub struct ComplexMatrixF32 {
    name: String,
    rows: usize,
    cols: usize,
    values: Vec<(f32, f32)>,
}

impl MatFile {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            variables: Vec::new(),
        }
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

    pub fn add_complex_f32_matrix(
        &mut self,
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        values: Vec<(f32, f32)>,
    ) -> Result<(), MatError> {
        let name = name.into();
        let expected = rows.saturating_mul(cols);
        if values.len() != expected {
            return Err(MatError::Shape {
                name,
                rows,
                cols,
                actual: values.len(),
            });
        }
        self.variables.push(MatVar::ComplexF32(ComplexMatrixF32 {
            name,
            rows,
            cols,
            values,
        }));
        Ok(())
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

fn matrix_with_values<T>(
    name: String,
    rows: usize,
    cols: usize,
    values: Vec<T>,
) -> Result<Matrix<T>, MatError> {
    let expected = rows.saturating_mul(cols);
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
        MatVar::F64(matrix) => {
            write_array_flags(&mut body, MX_DOUBLE_CLASS, false)?;
            write_dimensions(&mut body, matrix.rows, matrix.cols)?;
            write_name(&mut body, &matrix.name)?;
            write_numeric_f64(&mut body, matrix)?;
        }
        MatVar::F32(matrix) => {
            write_array_flags(&mut body, MX_SINGLE_CLASS, false)?;
            write_dimensions(&mut body, matrix.rows, matrix.cols)?;
            write_name(&mut body, &matrix.name)?;
            write_numeric_f32(&mut body, matrix)?;
        }
        MatVar::ComplexF32(matrix) => {
            write_array_flags(&mut body, MX_SINGLE_CLASS, true)?;
            write_dimensions(&mut body, matrix.rows, matrix.cols)?;
            write_name(&mut body, &matrix.name)?;
            write_complex_f32(matrix, &mut body)?;
        }
    }
    write_tag(file, MI_MATRIX, body.len())?;
    file.write_all(&body)?;
    pad_to_8(file, body.len())
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

fn write_dimensions(out: &mut Vec<u8>, rows: usize, cols: usize) -> io::Result<()> {
    write_tag(out, MI_INT32, 8)?;
    out.write_all(&(rows as i32).to_le_bytes())?;
    out.write_all(&(cols as i32).to_le_bytes())
}

fn write_name(out: &mut Vec<u8>, name: &str) -> io::Result<()> {
    write_tag(out, MI_INT8, name.len())?;
    out.write_all(name.as_bytes())?;
    pad_to_8(out, name.len())
}

fn write_numeric_f64(out: &mut Vec<u8>, matrix: &Matrix<f64>) -> io::Result<()> {
    let byte_len = matrix.values.len() * std::mem::size_of::<f64>();
    write_tag(out, MI_DOUBLE, byte_len)?;
    for col in 0..matrix.cols {
        for row in 0..matrix.rows {
            out.write_all(&matrix.values[row * matrix.cols + col].to_le_bytes())?;
        }
    }
    pad_to_8(out, byte_len)
}

fn write_numeric_f32(out: &mut Vec<u8>, matrix: &Matrix<f32>) -> io::Result<()> {
    let byte_len = matrix.values.len() * std::mem::size_of::<f32>();
    write_tag(out, MI_SINGLE, byte_len)?;
    for col in 0..matrix.cols {
        for row in 0..matrix.rows {
            out.write_all(&matrix.values[row * matrix.cols + col].to_le_bytes())?;
        }
    }
    pad_to_8(out, byte_len)
}

fn write_complex_f32(matrix: &ComplexMatrixF32, out: &mut Vec<u8>) -> io::Result<()> {
    let bytes = matrix.values.len() * std::mem::size_of::<f32>();
    write_tag(out, MI_SINGLE, bytes)?;
    for col in 0..matrix.cols {
        for row in 0..matrix.rows {
            out.write_all(&matrix.values[row * matrix.cols + col].0.to_le_bytes())?;
        }
    }
    pad_to_8(out, bytes)?;
    write_tag(out, MI_SINGLE, bytes)?;
    for col in 0..matrix.cols {
        for row in 0..matrix.rows {
            out.write_all(&matrix.values[row * matrix.cols + col].1.to_le_bytes())?;
        }
    }
    pad_to_8(out, bytes)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_shape_mismatch() {
        let mut mat = MatFile::new("unused.mat");
        let err = mat.add_f64_matrix("x", 2, 2, vec![1.0, 2.0, 3.0]).unwrap_err();
        assert!(err.to_string().contains("3 values for 2x2"));
    }

    #[test]
    fn writes_mat_v5_header() {
        let path = std::env::temp_dir().join(format!("pystamps-mat-{}.mat", std::process::id()));
        let mut mat = MatFile::new(&path);
        mat.add_f64_matrix("x", 1, 2, vec![1.0, 2.0]).unwrap();
        mat.write().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"MATLAB 5.0 MAT-file"));
        assert_eq!(&bytes[126..128], b"IM");
        std::fs::remove_file(path).unwrap();
    }
}
