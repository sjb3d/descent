use arrayvec::ArrayVec;
use std::{fmt, iter, mem, ops};

pub(crate) const MAX_DIM: usize = 4;
pub(crate) type ShapeVec = ArrayVec<isize, MAX_DIM>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Axis(u8);

impl Axis {
    pub(crate) fn from_index(index: usize) -> Axis {
        assert!(index < MAX_DIM);
        Self(index as u8)
    }

    pub(crate) fn index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(ShapeVec);

impl Shape {
    pub(crate) fn new(v: ShapeVec) -> Self {
        assert!(!v.is_empty());
        assert!(v.iter().all(|&a| a > 0));
        Self(v)
    }

    fn iter_rev_then_one<'s>(&'s self, len: usize) -> impl Iterator<Item = isize> + 's {
        self.0
            .iter()
            .copied()
            .rev()
            .chain(iter::repeat(1))
            .take(len)
    }

    pub(crate) fn match_with_broadcast(&self, rhs: &Shape) -> Self {
        // broadcast axes from 1 => n where necessary
        let len = self.0.len().max(rhs.0.len());
        let rev: ShapeVec = self
            .iter_rev_then_one(len)
            .zip(rhs.iter_rev_then_one(len))
            .map(|(a, b)| match (a, b) {
                (1, n) => n,
                (m, 1) => m,
                (m, n) => {
                    assert_eq!(m, n);
                    m
                }
            })
            .collect();
        Shape::new(rev.into_iter().rev().collect())
    }

    pub(crate) fn reduce_axis_onto_per_element(&self, rhs: &Shape) -> Option<isize> {
        if self.0.len() > rhs.0.len() {
            return Some(0);
        }
        assert_eq!(self.0.len(), rhs.0.len());
        for (i, (a, b)) in self
            .0
            .iter()
            .copied()
            .zip(rhs.0.iter().copied())
            .enumerate()
        {
            if a != b {
                assert_eq!(b, 1);
                return Some(i as isize);
            }
        }
        None
    }

    pub(crate) fn matrix_multiply(&self, rhs: &Shape) -> (Self, isize) {
        assert_eq!(rhs.0.len(), 2);
        let (a_last, a_prefix) = self.0.split_last().unwrap();
        let (b_first, b_suffix) = rhs.0.split_first().unwrap();
        assert_eq!(*a_last, *b_first);
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(a_prefix).unwrap();
        v.try_extend_from_slice(b_suffix).unwrap();
        (Shape::new(v), *a_last)
    }

    pub(crate) fn transposed(&self) -> Self {
        assert_eq!(self.0.len(), 2);
        Shape::new(self.0.iter().copied().rev().collect())
    }

    pub(crate) fn identity_view(&self) -> View {
        View::identity(self.0.len())
    }

    pub(crate) fn axis(&self, index: isize) -> Axis {
        // address from end if negative
        Axis::from_index((index as usize).wrapping_add(if index < 0 { self.0.len() } else { 0 }))
    }

    pub(crate) fn reduce(&self, axis: Axis) -> Self {
        // strip outermost dimension if reduced, otherwise keep with length 1
        let index = axis.index();
        if index == 0 {
            Shape::new(self.0.iter().copied().skip(1).collect())
        } else {
            let mut v = self.0.clone();
            v[index] = 1;
            Shape::new(v)
        }
    }

    pub(crate) fn one_hot(&self, count: isize) -> Self {
        // expand last axis (innermost dimension) from 1 to n
        let (last, prefix) = self.0.split_last().unwrap();
        assert_eq!(*last, 1);
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(prefix).unwrap();
        v.push(count);
        Shape::new(v)
    }

    fn strides(&self) -> Self {
        let mut stride = 1;
        let v: ShapeVec = self
            .0
            .iter()
            .copied()
            .rev()
            .map(|n| {
                let m = stride;
                stride *= n;
                m
            })
            .collect();
        Shape(v.iter().copied().rev().collect())
    }

    pub(crate) fn element_count(&self) -> usize {
        self.0.iter().copied().product::<isize>() as usize
    }

    pub(crate) fn buffer_size(&self) -> usize {
        self.element_count() * mem::size_of::<f32>()
    }
}

impl ops::Deref for Shape {
    type Target = [isize];
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_list(self.0.iter(), f)
    }
}

impl<const N: usize> From<[isize; N]> for Shape {
    fn from(s: [isize; N]) -> Self {
        Self::new(s.iter().copied().collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AxisRemap {
    step: isize,
    axis: Axis,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct View {
    offsets: ArrayVec<isize, MAX_DIM>,
    remap: ArrayVec<AxisRemap, MAX_DIM>,
}

impl View {
    fn identity(dims: usize) -> Self {
        Self {
            offsets: iter::repeat(0).take(dims).collect(),
            remap: (0..dims)
                .map(|index| AxisRemap {
                    step: 1,
                    axis: Axis::from_index(index),
                })
                .collect(),
        }
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.eq(&Self::identity(self.remap.len()))
    }

    pub(crate) fn through(&self, view: &View) -> Self {
        assert_eq!(self.remap.len(), view.offsets.len());
        let mut offsets = self.offsets.clone();
        for (remap, offset) in self.remap.iter().copied().zip(view.offsets.iter().copied()) {
            offsets[remap.axis.index()] += remap.step * offset;
        }
        let remap = view
            .remap
            .iter()
            .map(|outer| {
                let inner = self.remap[outer.axis.index()];
                AxisRemap {
                    step: outer.step * inner.step,
                    axis: inner.axis,
                }
            })
            .collect();
        Self { offsets, remap }
    }

    pub(crate) fn transposed(&self) -> Self {
        assert_eq!(self.remap.len(), 2);
        Self {
            offsets: self.offsets.clone(),
            remap: self.remap.iter().copied().rev().collect(),
        }
    }

    pub(crate) fn broadcast(from: &Shape, to: &Shape) -> Self {
        assert!(from.len() <= to.len());
        let offsets = iter::repeat(0).take(from.len()).collect();
        let mut remap = ArrayVec::new();
        while remap.len() + from.len() < to.len() {
            remap.push(AxisRemap {
                axis: Axis::from_index(0),
                step: 0,
            });
        }
        for (index, (&from, &to)) in from.iter().zip(to.iter().skip(remap.len())).enumerate() {
            remap.push(AxisRemap {
                axis: Axis::from_index(index),
                step: if from == to {
                    1
                } else {
                    assert_eq!(from, 1);
                    0
                },
            });
        }
        Self { offsets, remap }
    }
}

pub(crate) struct ViewIndexer {
    pub(crate) scales: ArrayVec<isize, MAX_DIM>,
    pub(crate) offset: isize,
}

impl ViewIndexer {
    pub(crate) fn new(shape: &Shape, view: &View) -> Self {
        assert_eq!(shape.len(), view.offsets.len());
        let strides = shape.strides();

        let scales = view
            .remap
            .iter()
            .map(|remap| strides[remap.axis.index()] * remap.step)
            .collect();
        let offset = view
            .offsets
            .iter()
            .cloned()
            .zip(strides.iter().cloned())
            .map(|(offset, stride)| offset * stride)
            .sum();

        Self { scales, offset }
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
