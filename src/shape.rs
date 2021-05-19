use arrayvec::ArrayVec;
use std::{fmt, iter};

pub(crate) type AxisLen = usize;
pub(crate) type ShapeVec = ArrayVec<AxisLen, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(ShapeVec);

impl Shape {
    pub(crate) fn iter_rev_then_one(&self, len: usize) -> impl Iterator<Item = &AxisLen> {
        self.0.iter().rev().chain(iter::repeat(&1)).take(len)
    }

    pub(crate) fn per_element(&self, rhs: &Shape) -> Self {
        // broadcast axes from 1 => n where necessary
        let len = self.0.len().max(rhs.0.len());
        let rev: ShapeVec = self
            .iter_rev_then_one(len)
            .zip(rhs.iter_rev_then_one(len))
            .map(|(&a, &b)| match (a, b) {
                (1, n) => n,
                (m, 1) => m,
                (m, n) => {
                    assert_eq!(m, n);
                    m
                }
            })
            .collect();
        Shape(rev.iter().cloned().rev().collect())
    }

    pub(crate) fn matrix_multiply(&self, rhs: &Shape) -> Self {
        assert_eq!(rhs.0.len(), 2);
        let (a_last, a_prefix) = self.0.split_last().unwrap();
        let (b_first, b_suffix) = rhs.0.split_first().unwrap();
        assert_eq!(*a_last, *b_first);
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(a_prefix).unwrap();
        v.try_extend_from_slice(b_suffix).unwrap();
        Shape(v)
    }

    pub(crate) fn transpose(&self) -> Self {
        assert_eq!(self.0.len(), 2);
        Shape(self.0.iter().cloned().rev().collect())
    }

    pub(crate) fn reduce(&self, axis: isize) -> Self {
        // address from end if negative
        let axis = (axis as usize).wrapping_add(if axis < 0 { self.0.len() } else { 0 });

        // strip outermost dimension if reduced, otherwise keep with length 1
        if axis == 0 {
            Shape(self.0.iter().cloned().skip(1).collect())
        } else {
            let mut v = self.0.clone();
            v[axis] = 1;
            Shape(v)
        }
    }

    pub(crate) fn one_hot(&self, count: AxisLen) -> Self {
        // expand last axis (innermost dimension) from 1 to n
        let (last, prefix) = self.0.split_last().unwrap();
        assert_eq!(*last, 1);
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(prefix).unwrap();
        v.push(count);
        Shape(v)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_list(self.0.iter(), f)
    }
}

impl<const N: usize> From<[AxisLen; N]> for Shape {
    fn from(s: [AxisLen; N]) -> Self {
        Self(s.iter().cloned().collect())
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
