use arrayvec::ArrayVec;
use std::{convert::TryInto, fmt, iter, mem, ops};

pub(crate) const MAX_DIM: usize = 4;
pub(crate) type ShapeVec = ArrayVec<usize, MAX_DIM>;

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

    pub(crate) fn prefix_ones_to_len(&self, len: usize) -> Self {
        let mut v = ShapeVec::new();
        while v.len() + self.len() < len {
            v.push(1);
        }
        v.try_extend_from_slice(self).unwrap();
        Shape::new(v)
    }

    pub(crate) fn match_with_broadcast(&self, rhs: &Shape) -> Self {
        // broadcast axes from 1 => n where necessary
        let len = self.0.len().max(rhs.0.len());
        let a = self.prefix_ones_to_len(len);
        let b = rhs.prefix_ones_to_len(len);
        Shape::new(
            a.iter()
                .copied()
                .zip(b.iter().copied())
                .map(|(a, b)| match (a, b) {
                    (1, n) => n,
                    (m, 1) => m,
                    (m, n) => {
                        assert_eq!(m, n);
                        m
                    }
                })
                .collect(),
        )
    }

    pub(crate) fn reduce_axis_onto_per_element(&self, rhs: &Shape) -> Option<Axis> {
        if self.0.len() > rhs.0.len() {
            return Some(Axis::from_index(0));
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
                return Some(Axis::from_index(i));
            }
        }
        None
    }

    pub(crate) fn matrix_multiply(&self, rhs: &Shape) -> Self {
        assert_eq!(rhs.0.len(), 2);
        let (a_last, a_prefix) = self.0.split_last().unwrap();
        let (b_first, b_suffix) = rhs.0.split_first().unwrap();
        assert_eq!(*a_last, *b_first);
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(a_prefix).unwrap();
        v.try_extend_from_slice(b_suffix).unwrap();
        Shape::new(v)
    }

    pub(crate) fn conv2d(&self, filters: &Shape, pad: usize) -> Self {
        let [n, in_c, in_h, in_w]: [usize; 4] = self.0.as_slice().try_into().unwrap();
        let [out_c, fc, fh, fw]: [usize; 4] = filters.0.as_slice().try_into().unwrap();
        assert_eq!(in_c, fc);
        let out_w = 1 + in_w + 2 * pad - fw;
        let out_h = 1 + in_h + 2 * pad - fh;
        Shape::from([n, out_c, out_h, out_w])
    }

    pub(crate) fn transposed(&self) -> Self {
        assert_eq!(self.0.len(), 2);
        Shape::new(self.0.iter().copied().rev().collect())
    }

    pub(crate) fn identity_view(&self) -> View {
        View::identity(self)
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

    pub(crate) fn one_hot(&self, count: usize) -> Self {
        // expand last axis (innermost dimension) from 1 to n
        let (last, prefix) = self.0.split_last().unwrap();
        assert_eq!(*last, 1);
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(prefix).unwrap();
        v.push(count);
        Shape::new(v)
    }

    fn strides(&self) -> ArrayVec<isize, MAX_DIM> {
        let mut stride = 1;
        let v: ArrayVec<isize, MAX_DIM> = self
            .0
            .iter()
            .copied()
            .rev()
            .map(|n| {
                let m = stride;
                stride *= n;
                if n > 1 {
                    m as isize
                } else {
                    0
                }
            })
            .collect();
        v.iter().copied().rev().collect()
    }

    pub fn element_count(&self) -> usize {
        self.0.iter().copied().product::<usize>() as usize
    }

    pub(crate) fn buffer_size(&self) -> usize {
        self.element_count() * mem::size_of::<f32>()
    }
}

impl ops::Deref for Shape {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_list(self.0.iter(), f)
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(s: [usize; N]) -> Self {
        Self::new(s.iter().copied().collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ViewSource {
    stride: usize,
    offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ViewMapping {
    Source { axis: Axis, step: isize },
    Broadcast,
}

impl ViewMapping {
    fn identity(axis: Axis, length: usize) -> Self {
        if length > 1 {
            Self::Source { axis, step: 1 }
        } else {
            Self::Broadcast
        }
    }

    fn stepped(&self, step_multiplier: isize) -> Self {
        match *self {
            Self::Source { axis, step } => Self::Source {
                axis,
                step: step * step_multiplier,
            },
            Self::Broadcast => Self::Broadcast,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct View {
    input_shape: Shape,
    input_offsets: ArrayVec<isize, MAX_DIM>,
    output_mapping: ArrayVec<ViewMapping, MAX_DIM>,
    pub(crate) output_shape: Shape,
}

impl View {
    fn identity(shape: &Shape) -> Self {
        Self {
            input_shape: shape.clone(),
            input_offsets: iter::repeat(0).take(shape.len()).collect(),
            output_mapping: shape
                .iter()
                .copied()
                .enumerate()
                .map(|(index, shape)| ViewMapping::identity(Axis::from_index(index), shape))
                .collect(),
            output_shape: shape.clone(),
        }
    }

    pub(crate) fn is_identity(&self) -> bool {
        Self::identity(&self.output_shape).eq(self)
    }

    pub(crate) fn through(&self, view: &View) -> Self {
        assert_eq!(&self.output_shape, &view.input_shape);
        let mut input_offsets = self.input_offsets.clone();
        for (mapping, offset) in self
            .output_mapping
            .iter()
            .copied()
            .zip(view.input_offsets.iter().copied())
        {
            match mapping {
                ViewMapping::Source { axis, step } => input_offsets[axis.index()] += step * offset,
                ViewMapping::Broadcast => {}
            }
        }
        let output_mapping = view
            .output_mapping
            .iter()
            .copied()
            .map(|outer| match outer {
                ViewMapping::Source { axis, step } => {
                    self.output_mapping[axis.index()].stepped(step)
                }
                ViewMapping::Broadcast => ViewMapping::Broadcast,
            })
            .collect();
        Self {
            input_shape: self.input_shape.clone(),
            input_offsets,
            output_mapping,
            output_shape: view.output_shape.clone(),
        }
    }

    pub(crate) fn transposed(&self) -> Self {
        assert_eq!(self.output_mapping.len(), 2);
        Self {
            input_shape: self.input_shape.clone(),
            input_offsets: self.input_offsets.clone(),
            output_mapping: self.output_mapping.iter().copied().rev().collect(),
            output_shape: self.output_shape.transposed(),
        }
    }

    pub(crate) fn broadcast(input_shape: &Shape, output_shape: &Shape) -> Self {
        assert!(input_shape.len() <= output_shape.len());
        let input_offsets = iter::repeat(0).take(input_shape.len()).collect();
        let mut output_mapping = ArrayVec::new();
        while output_mapping.len() + input_shape.len() < output_shape.len() {
            output_mapping.push(ViewMapping::Broadcast);
        }
        for (index, (&from, &to)) in input_shape
            .iter()
            .zip(output_shape.iter().skip(output_mapping.len()))
            .enumerate()
        {
            output_mapping.push(if from == to {
                ViewMapping::identity(Axis::from_index(index), from)
            } else {
                ViewMapping::Broadcast
            });
        }
        Self {
            input_shape: input_shape.clone(),
            input_offsets,
            output_mapping,
            output_shape: output_shape.clone(),
        }
    }

    pub(crate) fn indexer(&self) -> ViewIndexer {
        ViewIndexer::new(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ViewIndexer {
    pub(crate) scales: ArrayVec<isize, MAX_DIM>,
    pub(crate) offset: isize,
}

impl ViewIndexer {
    fn new(view: &View) -> Self {
        let input_strides = view.input_shape.strides();

        let scales = view
            .output_mapping
            .iter()
            .map(|mapping| match mapping {
                ViewMapping::Source { axis, step } => input_strides[axis.index()] * step,
                ViewMapping::Broadcast => 0,
            })
            .collect();
        let offset = view
            .input_offsets
            .iter()
            .cloned()
            .zip(input_strides.iter().cloned())
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
