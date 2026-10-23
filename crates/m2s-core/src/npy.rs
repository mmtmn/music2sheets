//! Minimal reader for the `.npy` arrays the Python worker writes (2-D, little-endian float16/32/64).

use std::path::Path;

use anyhow::{bail, Context, Result};

/// A dense row-major 2-D array of f32 (frames x bins).
#[derive(Debug, Clone, PartialEq)]
pub struct Array2 {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Array2 {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Array2 {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
    /// One column as a vector (e.g. the beat activation out of [beat, downbeat]).
    pub fn column(&self, c: usize) -> Vec<f32> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }
}

pub fn read(path: impl AsRef<Path>) -> Result<Array2> {
    let p = path.as_ref();
    let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    parse(&bytes).with_context(|| format!("parsing {}", p.display()))
}

pub fn parse(bytes: &[u8]) -> Result<Array2> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not an .npy file");
    }
    let major = bytes[6];
    let (hlen, start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen])?;
    let descr = field(header, "descr").context("no descr")?;
    if header.contains("'fortran_order': True") {
        bail!("fortran order not supported");
    }
    let shape_txt = header.split("'shape':").nth(1).context("no shape")?;
    let shape_txt = &shape_txt[shape_txt.find('(').context("shape")? + 1..shape_txt.find(')').context("shape")?];
    let dims: Vec<usize> = shape_txt
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap_or(0))
        .collect();
    let (rows, cols) = match dims.as_slice() {
        [r] => (*r, 1),
        [r, c] => (*r, *c),
        _ => bail!("only 1-D or 2-D arrays are supported, got {dims:?}"),
    };
    let body = &bytes[start + hlen..];
    let n = rows * cols;
    let data: Vec<f32> = match descr.as_str() {
        "<f2" => body
            .chunks_exact(2)
            .take(n)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        "<f4" => body
            .chunks_exact(4)
            .take(n)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "<f8" => body
            .chunks_exact(8)
            .take(n)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        d => bail!("unsupported dtype {d}"),
    };
    if data.len() != n {
        bail!("truncated array: {} of {n} values", data.len());
    }
    Ok(Array2 { rows, cols, data })
}

fn field(header: &str, key: &str) -> Option<String> {
    let i = header.find(&format!("'{key}':"))?;
    let rest = &header[i + key.len() + 3..];
    let a = rest.find('\'')? + 1;
    let b = rest[a..].find('\'')? + a;
    Some(rest[a..b].to_string())
}

/// Write a float32 2-D array (used for cached activation cells).
pub fn write(path: impl AsRef<Path>, a: &Array2) -> Result<()> {
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        a.rows, a.cols
    );
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + a.data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in &a.data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path.as_ref(), out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let a = Array2 {
            rows: 2,
            cols: 3,
            data: vec![0.0, 1.5, -2.0, 3.25, 4.0, 5.0],
        };
        let dir = std::env::temp_dir().join("m2s_npy_test.npy");
        write(&dir, &a).unwrap();
        assert_eq!(read(&dir).unwrap(), a);
    }
}
