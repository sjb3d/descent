use matrixmultiply::sgemm;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use std::{
    cmp, fmt,
    iter::FromIterator,
    mem::MaybeUninit,
    ops,
    slice::{self, SliceIndex},
};

const MAX_DIMS: usize = 4;

#[derive(Clone, Copy)]
struct CopyableArrayVec<T: Copy, const N: usize> {
    elements: [T; N],
    len: usize,
}

impl<T: Copy, const N: usize> CopyableArrayVec<T, N> {
    pub fn new() -> Self {
        Self {
            elements: unsafe { MaybeUninit::zeroed().assume_init() },
            len: 0,
        }
    }

    pub fn push(&mut self, a: T) {
        self.elements[self.len] = a;
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[T] {
        &self.elements[..self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.elements[..self.len]
    }

    pub fn iter(&self) -> slice::Iter<T> {
        self.elements[..self.len].iter()
    }
}

impl<T: Copy + fmt::Debug, const N: usize> fmt::Debug for CopyableArrayVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", self.as_slice()))
    }
}

impl<T: Copy + PartialEq, const N: usize> cmp::PartialEq for CopyableArrayVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<T: Copy + Eq, const N: usize> cmp::Eq for CopyableArrayVec<T, N> {}

impl<T: Copy, const N: usize> ops::Deref for CopyableArrayVec<T, N> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy, const N: usize> ops::DerefMut for CopyableArrayVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Copy, const N: usize> FromIterator<T> for CopyableArrayVec<T, N> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut v = Self::new();
        for a in iter.into_iter() {
            v.push(a);
        }
        v
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size(CopyableArrayVec<usize, MAX_DIMS>);

impl Size {
    pub fn dims(&self) -> usize {
        self.0.len()
    }

    pub fn elements(&self) -> usize {
        self.0.iter().product()
    }

    fn split_first(&self) -> Option<(usize, Self)> {
        self.0
            .split_first()
            .map(|(first, remain)| (*first, remain.iter().cloned().collect()))
    }

    fn as_1d(&self) -> usize {
        assert_eq!(self.0.len(), 1);
        *self.0.first().unwrap()
    }

    fn as_2d(&self) -> (usize, usize) {
        assert_eq!(self.0.len(), 2);
        (*self.0.first().unwrap(), *self.0.last().unwrap())
    }

    fn from_add(a: &Size, b: &Size) -> Self {
        assert_eq!(a.dims(), b.dims());
        a.0.iter()
            .cloned()
            .zip(b.0.iter().cloned())
            .map(|(a, b)| a.max(b))
            .collect()
    }

    fn from_matrix_multiply(a: &Size, b: &Size) -> Self {
        assert_eq!(b.dims(), 2);
        assert_eq!(a.0.last(), b.0.first());
        let mut c = a.clone();
        *c.0.last_mut().unwrap() = *b.0.last().unwrap();
        c
    }

    fn packed_stride(&self) -> Stride {
        let mut tmp = CopyableArrayVec::<usize, MAX_DIMS>::new();
        tmp.push(1);
        for size in self[1..].iter().rev() {
            tmp.push(size * tmp.last().unwrap());
        }
        tmp.iter().rev().map(|n| *n as isize).collect()
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_list(self.0.iter(), f)
    }
}

impl<I: SliceIndex<[usize]>> ops::Index<I> for Size {
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl FromIterator<usize> for Size {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        Self(iter.into_iter().collect())
    }
}

impl<const N: usize> From<[usize; N]> for Size {
    fn from(arr: [usize; N]) -> Self {
        arr.iter().cloned().collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct Stride(CopyableArrayVec<isize, MAX_DIMS>);

impl Stride {
    fn as_1d(&self) -> isize {
        assert_eq!(self.0.len(), 1);
        *self.0.first().unwrap()
    }

    fn as_2d(&self) -> (isize, isize) {
        assert_eq!(self.0.len(), 2);
        (*self.0.first().unwrap(), *self.0.last().unwrap())
    }

    fn split_first(&self) -> Option<(isize, Self)> {
        self.0
            .split_first()
            .map(|(first, remain)| (*first, remain.iter().cloned().collect()))
    }
}

impl FromIterator<isize> for Stride {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = isize>,
    {
        Self(iter.into_iter().collect())
    }
}

#[derive(Debug)]
pub struct Array {
    elements: Vec<f32>,
    size: Size,
}

impl Array {
    pub fn from_elements(elements: Vec<f32>, size: impl Into<Size>) -> Self {
        let size = size.into();
        assert_eq!(elements.len(), size.elements());
        Self { elements, size }
    }

    pub fn zeros(size: impl Into<Size>) -> Self {
        let size = size.into();
        let elements = vec![0.0; size.elements()];
        Self { elements, size }
    }

    pub fn xavier_uniform(size: impl Into<Size>, rng: &mut impl Rng) -> Self {
        let size = size.into();
        let a = (6.0 / (size[0] as f32)).sqrt();
        let dist = Uniform::new(-a, a);
        Self {
            elements: dist.sample_iter(rng).take(size.elements()).collect(),
            size,
        }
    }

    pub fn size(&self) -> &Size {
        &self.size
    }

    pub fn view(&self) -> ArrayView {
        ArrayView {
            elements: &self.elements,
            offset: 0,
            stride: self.size.packed_stride(),
            size: self.size.clone(),
        }
    }

    pub fn view_mut(&mut self) -> ArrayViewMut {
        ArrayViewMut {
            elements: &mut self.elements,
            offset: 0,
            stride: self.size.packed_stride(),
            size: self.size.clone(),
        }
    }
}

impl<'a> ops::Add<&Array> for Array {
    type Output = Array;
    fn add(self, rhs: &Array) -> Self::Output {
        let add_size = Size::from_add(&self.size, &rhs.size);
        let mut c = Array::zeros(add_size);
        array_add(
            &self.view().broadcast(add_size),
            &rhs.view().broadcast(add_size),
            c.view_mut(),
        );
        c
    }
}

impl<'a> ops::Mul for &Array {
    type Output = Array;
    fn mul(self, rhs: &Array) -> Self::Output {
        // TODO: uninitialized c
        let mut c = Array::zeros(Size::from_matrix_multiply(&self.size, &rhs.size));
        matrix_multiply(1.0, &self.view(), &rhs.view(), 0.0, c.view_mut());
        c
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{}", self.view()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ArrayView<'a> {
    elements: &'a [f32],
    offset: isize,
    stride: Stride,
    size: Size,
}

#[derive(Debug)]
pub struct ArrayViewMut<'a> {
    elements: &'a mut [f32],
    offset: isize,
    stride: Stride,
    size: Size,
}

impl<'a> ArrayView<'a> {
    fn inner_iter(&self) -> impl Iterator<Item = ArrayView> {
        let (outer_size, inner_size) = self.size.split_first().unwrap();
        let (outer_stride, inner_stride) = self.stride.split_first().unwrap();
        (0..outer_size).map({
            move |outer_index| Self {
                elements: self.elements,
                offset: self.offset + (outer_index as isize) * outer_stride,
                stride: inner_stride,
                size: inner_size,
            }
        })
    }

    pub fn broadcast(&self, size: impl Into<Size>) -> Self {
        let size = size.into();
        let stride = self
            .stride
            .0
            .iter()
            .zip(self.size.0.iter())
            .zip(size.0.iter())
            .map(|((&old_stride, &old_size), &new_size)| {
                if old_size == new_size {
                    old_stride
                } else {
                    assert_eq!(old_size, 1);
                    0
                }
            })
            .collect();
        Self {
            elements: self.elements,
            offset: self.offset,
            stride,
            size,
        }
    }
}

impl<'a> ArrayViewMut<'a> {
    fn into_inner_iter(self) -> impl Iterator<Item = ArrayViewMut<'a>> {
        let (outer_size, inner_size) = self.size.split_first().unwrap();
        let (outer_stride, inner_stride) = self.stride.split_first().unwrap();
        (0..outer_size).map({
            let elements_ptr = self.elements.as_mut_ptr();
            let elements_len = self.elements.len();
            move |outer_index| Self {
                // SAFETY: this view is consumed, inner views are iterated one at a time
                elements: unsafe { slice::from_raw_parts_mut(elements_ptr, elements_len) },
                offset: self.offset + (outer_index as isize) * outer_stride,
                stride: inner_stride,
                size: inner_size,
            }
        })
    }
}

fn display_list<I>(iter: I, f: &mut fmt::Formatter<'_>) -> fmt::Result
where
    I: IntoIterator,
    I::Item: fmt::Display,
{
    let mut iter = iter.into_iter().peekable();
    f.write_str("[")?;
    while let Some(x) = iter.next() {
        f.write_fmt(format_args!("{}", x))?;
        if iter.peek().is_some() {
            f.write_str(", ")?;
        }
    }
    f.write_str("]")
}

impl<'a> fmt::Display for ArrayView<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.size.dims() == 0 {
            f.write_fmt(format_args!("{}", self.elements[self.offset as usize]))
        } else {
            display_list(self.inner_iter(), f)
        }
    }
}

fn array_add(a: &ArrayView, b: &ArrayView, c: ArrayViewMut) {
    assert_eq!(a.size, b.size);
    assert_eq!(a.size, c.size);
    // TODO: reshape to optimise when contiguous?
    array_add_impl(a, b, c);
}

fn array_add_impl(a: &ArrayView, b: &ArrayView, c: ArrayViewMut) {
    if a.size.dims() == 1 {
        let n = a.size.as_1d();
        let sa = a.stride.as_1d();
        let sb = b.stride.as_1d();
        let sc = c.stride.as_1d();
        unsafe {
            let mut a_ptr = a.elements.as_ptr().offset(a.offset);
            let mut b_ptr = b.elements.as_ptr().offset(b.offset);
            let mut c_ptr = c.elements.as_mut_ptr().offset(c.offset);
            for _ in 0..n {
                *c_ptr = *a_ptr + *b_ptr;
                a_ptr = a_ptr.offset(sa);
                b_ptr = b_ptr.offset(sb);
                c_ptr = c_ptr.offset(sc);
            }
        }
    } else {
        for ((a, b), c) in a.inner_iter().zip(b.inner_iter()).zip(c.into_inner_iter()) {
            array_add_impl(&a, &b, c);
        }
    }
}

fn matrix_multiply(alpha: f32, a: &ArrayView, b: &ArrayView, beta: f32, c: ArrayViewMut) {
    assert_eq!(b.size.dims(), 2);
    assert_eq!(a.size.0.last(), b.size.0.first());
    assert_eq!(c.size, Size::from_matrix_multiply(&a.size, &b.size));
    matrix_multiply_impl(alpha, a, b, beta, c);
}

fn matrix_multiply_impl(alpha: f32, a: &ArrayView, b: &ArrayView, beta: f32, c: ArrayViewMut) {
    if a.size.dims() == 2 {
        let (m, _) = a.size.as_2d();
        let (k, n) = b.size.as_2d();
        let (rsa, csa) = a.stride.as_2d();
        let (rsb, csb) = b.stride.as_2d();
        let (rsc, csc) = c.stride.as_2d();
        unsafe {
            let a_ptr = a.elements.as_ptr().offset(a.offset);
            let b_ptr = b.elements.as_ptr().offset(b.offset);
            let c_ptr = c.elements.as_mut_ptr().offset(c.offset);
            sgemm(
                m, k, n, alpha, a_ptr, rsa, csa, b_ptr, rsb, csb, beta, c_ptr, rsc, csc,
            );
        }
    } else {
        for (a, c) in a.inner_iter().zip(c.into_inner_iter()) {
            matrix_multiply(alpha, &a, b, beta, c);
        }
    }
}
