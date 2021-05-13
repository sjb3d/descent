use arrayvec::ArrayVec;
use matrixmultiply::sgemm;
use std::{cell::UnsafeCell, fmt, rc::Rc};

const MAX_DIMS: usize = 4;

#[derive(Debug, Clone)]
pub struct Size(ArrayVec<usize, MAX_DIMS>);

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
            .map(|(first, remain)| (*first, Self(remain.iter().cloned().collect())))
    }

    fn as_1d(&self) -> usize {
        assert_eq!(self.0.len(), 1);
        *self.0.first().unwrap()
    }

    fn as_2d(&self) -> (usize, usize) {
        assert_eq!(self.0.len(), 2);
        (*self.0.first().unwrap(), *self.0.last().unwrap())
    }

    fn matrix_multiply(a: &Size, b: &Size) -> Size {
        assert_eq!(b.dims(), 2);
        assert_eq!(a.0.last(), b.0.first());
        let mut c = a.clone();
        *c.0.last_mut().unwrap() = *b.0.last().unwrap();
        c
    }

    fn packed_stride(&self) -> Stride {
        let mut tmp = ArrayVec::<usize, MAX_DIMS>::new();
        tmp.push(1);
        for size in self.0[1..].iter().rev() {
            tmp.push(size * tmp.last().unwrap());
        }
        Stride(tmp.iter().rev().map(|n| *n as isize).collect())
    }
}

#[derive(Debug, Clone)]
struct Stride(ArrayVec<isize, MAX_DIMS>);

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
            .map(|(first, remain)| (*first, Self(remain.iter().cloned().collect())))
    }
}

impl<const N: usize> Into<Size> for [usize; N] {
    fn into(self) -> Size {
        Size(self.iter().cloned().collect())
    }
}

#[derive(Debug, Clone)]
struct RcStore(Rc<UnsafeCell<Vec<f32>>>);

impl RcStore {
    fn from_elements(elements: Vec<f32>) -> Self {
        Self(Rc::new(UnsafeCell::new(elements)))
    }
}

pub struct Array {
    store: RcStore,
    offset: isize,
    stride: Stride,
    size: Size,
}

impl Array {
    pub fn from_elements(elements: Vec<f32>, size: impl Into<Size>) -> Self {
        let size = size.into();
        assert_eq!(elements.len(), size.elements());
        Self {
            store: RcStore::from_elements(elements),
            offset: 0,
            stride: size.packed_stride(),
            size,
        }
    }

    pub fn zeros(size: impl Into<Size>) -> Self {
        let size = size.into();
        let elements = vec![0.0; size.elements()];
        Self::from_elements(elements, size)
    }

    pub fn size(&self) -> &Size {
        &self.size
    }

    fn iter_inner(&self) -> impl Iterator<Item = Array> {
        let (outer_size, inner_size) = self.size.split_first().unwrap();
        let (outer_stride, inner_stride) = self.stride.split_first().unwrap();
        (0..outer_size).map({
            let rc_store = self.store.clone();
            let offset = self.offset;
            move |outer_index| Array {
                store: rc_store.clone(),
                offset: offset + (outer_index as isize) * outer_stride,
                stride: inner_stride.clone(),
                size: inner_size.clone(),
            }
        })
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.size.dims() == 1 {
            let elements = unsafe { self.store.0.get().as_ref().unwrap() };
            fmt.debug_list()
                .entries(
                    elements
                        .iter()
                        .skip(self.offset as usize)
                        .step_by(self.stride.as_1d() as usize)
                        .take(self.size.as_1d()),
                )
                .finish()
        } else {
            fmt.debug_list().entries(self.iter_inner()).finish()
        }
    }
}

pub fn matrix_multiply(alpha: f32, a: &Array, b: &Array) -> Array {
    let c = Array::zeros(Size::matrix_multiply(&a.size, &b.size));

    // TODO: iterate 2D blocks of A/C
    assert_eq!(a.size.dims(), 2);

    let (m, _) = a.size.as_2d();
    let (k, n) = b.size.as_2d();
    let (csa, rsa) = a.stride.as_2d();
    let (csb, rsb) = b.stride.as_2d();
    let (csc, rsc) = c.stride.as_2d();
    unsafe {
        let a_ptr = a.store.0.get().as_ref().unwrap().as_ptr().offset(a.offset);
        let b_ptr = b.store.0.get().as_ref().unwrap().as_ptr().offset(b.offset);
        let c_ptr = c
            .store
            .0
            .get()
            .as_mut()
            .unwrap()
            .as_mut_ptr()
            .offset(c.offset);
        sgemm(
            m, k, n, alpha, a_ptr, rsa, csa, b_ptr, rsb, csb, 0.0, c_ptr, rsc, csc,
        );
    }

    c
}
