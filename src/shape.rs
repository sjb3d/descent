use crate::common::*;
use std::{
    array,
    convert::{TryFrom, TryInto},
    fmt, iter, mem, ops,
    slice::SliceIndex,
};
use tinyvec::ArrayVec as TinyVec;

pub(crate) const MAX_DIM: usize = 7;
pub(crate) type ShapeVec = TinyVec<[usize; MAX_DIM]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Axis(u8);

impl Axis {
    pub(crate) fn from_index(index: usize) -> Self {
        assert!(index < MAX_DIM);
        Self(index as u8)
    }

    pub(crate) fn index(&self) -> usize {
        self.0 as usize
    }
}

pub(crate) trait DivRoundUp {
    fn div_round_up(self, x: Self) -> Self;
}

impl DivRoundUp for usize {
    fn div_round_up(self, x: Self) -> Self {
        (self + x - 1) / x
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignedIndex(pub isize);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape(ShapeVec);

impl Shape {
    pub(crate) fn new(v: ShapeVec) -> Self {
        assert!(!v.is_empty());
        assert!(v.iter().all(|&a| a > 0));
        Self(v)
    }

    pub(crate) fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [usize] {
        self.0.as_mut_slice()
    }

    pub(crate) fn rsplit_at(&self, rmid: usize) -> (&[usize], &[usize]) {
        self.0.split_at(self.0.len() - rmid)
    }

    pub(crate) fn prefix_ones_to_len(&self, len: usize) -> Self {
        let mut v = ShapeVec::new();
        while v.len() + self.len() < len {
            v.push(1);
        }
        v.extend_from_slice(self);
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

    pub(crate) fn matmul(&self, rhs: &Shape) -> Self {
        let [m, k0]: [usize; 2] = self.try_into().unwrap();
        let [k1, n]: [usize; 2] = rhs.try_into().unwrap();
        assert_eq!(k0, k1);
        let r = k0.div_round_up(MATMUL_MAX_K_SIZE);
        Shape::from([r, m, n])
    }

    pub(crate) fn pad_image(&self, pad: usize) -> Self {
        let mut p = *self;
        p[SignedIndex(-3)] += 2 * pad;
        p[SignedIndex(-2)] += 2 * pad;
        p
    }

    pub(crate) fn image_to_windows(
        &self,
        filter: (usize, usize),
        stride: (usize, usize),
        groups: usize,
    ) -> Self {
        assert!(self.0.len() >= 3);
        let (prefix, suffix) = self.rsplit_at(3);
        let [in_h, in_w, in_nc]: [usize; 3] = suffix.try_into().unwrap();
        assert_eq!(in_nc % groups, 0);
        let group_nc = in_nc / groups;
        let (filter_w, filter_h) = filter;
        let (stride_w, stride_h) = stride;
        let out_w = (in_w - filter_w) / stride_w + 1;
        let out_h = (in_h - filter_h) / stride_h + 1;
        assert_eq!((out_w - 1) * stride_w, in_w - filter_w);
        assert_eq!((out_h - 1) * stride_h, in_h - filter_h);
        let mut v = ShapeVec::new();
        v.extend_from_slice(prefix);
        v.extend_from_slice(&[out_h, out_w, groups, filter_h, filter_w, group_nc]);
        Shape::new(v)
    }

    pub(crate) fn windows_to_image(&self, stride: (usize, usize)) -> Self {
        assert!(self.0.len() >= 6);
        let (prefix, suffix) = self.rsplit_at(6);
        let [out_h, out_w, groups, filter_h, filter_w, group_nc]: [usize; 6] =
            suffix.try_into().unwrap();
        let (stride_w, stride_h) = stride;
        let in_nc = groups * group_nc;
        let in_w = (out_w - 1) * stride_w + filter_w;
        let in_h = (out_h - 1) * stride_h + filter_h;
        let mut v = ShapeVec::new();
        v.extend_from_slice(prefix);
        v.extend_from_slice(&[in_h, in_w, in_nc]);
        Shape::new(v)
    }

    pub(crate) fn transposed(&self) -> Self {
        let [a, b]: [usize; 2] = self.try_into().unwrap();
        Shape::from([b, a])
    }

    pub(crate) fn identity_view(&self) -> View {
        View::new(self)
    }

    pub(crate) fn padded_view(&self, pad: &[usize]) -> View {
        View::with_pad(self, pad)
    }

    pub(crate) fn axis(&self, index: isize) -> Axis {
        // address from end if negative
        Axis::from_index((index as usize).wrapping_add(if index < 0 { self.0.len() } else { 0 }))
    }

    pub(crate) fn reduce(&self, axis: Axis) -> Self {
        // strip outermost dimension if reduced, otherwise keep with length 1
        let index = axis.index();
        if index == 0 && self.0.len() > 1 {
            Shape::new(self.0.iter().copied().skip(1).collect())
        } else {
            let mut v = self.0;
            v[index] = 1;
            Shape::new(v)
        }
    }

    pub(crate) fn one_hot(&self, count: usize) -> Self {
        // expand last axis (innermost dimension) from 1 to n
        let (last, prefix) = self.0.split_last().unwrap();
        assert_eq!(*last, 1);
        let mut v = ShapeVec::new();
        v.extend_from_slice(prefix);
        v.push(count);
        Shape::new(v)
    }

    pub(crate) fn strides(&self) -> TinyVec<[usize; MAX_DIM]> {
        let mut stride = 1;
        let v: TinyVec<[usize; MAX_DIM]> = self
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

impl<const N: usize> TryFrom<Shape> for [usize; N] {
    type Error = array::TryFromSliceError;
    fn try_from(value: Shape) -> Result<Self, Self::Error> {
        value.as_slice().try_into()
    }
}
impl<const N: usize> TryFrom<&Shape> for [usize; N] {
    type Error = array::TryFromSliceError;
    fn try_from(value: &Shape) -> Result<Self, Self::Error> {
        value.as_slice().try_into()
    }
}

impl<I: SliceIndex<[usize]>> ops::Index<I> for Shape {
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}
impl<I: SliceIndex<[usize]>> ops::IndexMut<I> for Shape {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl ops::Index<Axis> for Shape {
    type Output = usize;
    fn index(&self, axis: Axis) -> &Self::Output {
        self.index(axis.index())
    }
}
impl ops::IndexMut<Axis> for Shape {
    fn index_mut(&mut self, axis: Axis) -> &mut Self::Output {
        self.index_mut(axis.index())
    }
}

impl ops::Index<SignedIndex> for Shape {
    type Output = usize;
    fn index(&self, index: SignedIndex) -> &Self::Output {
        self.index(self.axis(index.0))
    }
}
impl ops::IndexMut<SignedIndex> for Shape {
    fn index_mut(&mut self, index: SignedIndex) -> &mut Self::Output {
        self.index_mut(self.axis(index.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AxisMapping {
    Source { axis: Axis, step: isize },
    Broadcast,
}

impl Default for AxisMapping {
    fn default() -> Self {
        Self::Broadcast
    }
}

impl AxisMapping {
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct View {
    pub(crate) input_shape: Shape,
    pub(crate) input_padding: TinyVec<[usize; MAX_DIM]>,
    pub(crate) input_offsets: TinyVec<[isize; MAX_DIM]>,
    pub(crate) output_mapping: TinyVec<[AxisMapping; MAX_DIM]>,
    pub(crate) output_shape: Shape,
}

impl View {
    fn new(shape: &Shape) -> Self {
        Self {
            input_shape: *shape,
            input_padding: iter::repeat(0).take(shape.len()).collect(),
            input_offsets: iter::repeat(0).take(shape.len()).collect(),
            output_mapping: shape
                .iter()
                .copied()
                .enumerate()
                .map(|(index, len)| AxisMapping::identity(Axis::from_index(index), len))
                .collect(),
            output_shape: *shape,
        }
    }

    fn with_pad(shape: &Shape, pad: &[usize]) -> Self {
        assert_eq!(shape.len(), pad.len());
        let padded_shape = Shape::new(
            shape
                .iter()
                .copied()
                .zip(pad.iter().copied())
                .map(|(n, pad)| n + 2 * pad)
                .collect(),
        );
        Self {
            input_shape: *shape,
            input_padding: pad.iter().cloned().collect(),
            input_offsets: iter::repeat(0).take(shape.len()).collect(),
            output_mapping: shape
                .iter()
                .copied()
                .enumerate()
                .map(|(index, len)| AxisMapping::identity(Axis::from_index(index), len))
                .collect(),
            output_shape: padded_shape,
        }
    }

    pub(crate) fn is_identity(&self) -> bool {
        Self::new(&self.output_shape).eq(self)
    }

    fn can_restrict_to(&self, view: &View) -> bool {
        self.output_shape == view.input_shape
            && view.input_padding.iter().all(|&padding| padding == 0)
    }

    fn can_reshape_to(&self, view: &View) -> bool {
        self.is_identity() && self.output_shape.element_count() == view.input_shape.element_count()
    }

    pub(crate) fn can_view_through(&self, view: &View, can_reshape: bool) -> bool {
        self.can_restrict_to(view) || (can_reshape && self.can_reshape_to(view))
    }

    pub(crate) fn through(&self, view: &View, can_reshape: bool) -> Self {
        if !self.can_restrict_to(view) {
            assert!(can_reshape && self.can_reshape_to(view));
            return *view;
        }

        let mut input_offsets = self.input_offsets;
        for (mapping, offset) in self
            .output_mapping
            .iter()
            .copied()
            .zip(view.input_offsets.iter().copied())
        {
            match mapping {
                AxisMapping::Source { axis, step } => input_offsets[axis.index()] += step * offset,
                AxisMapping::Broadcast => {}
            }
        }
        let output_mapping = view
            .output_mapping
            .iter()
            .copied()
            .map(|outer| match outer {
                AxisMapping::Source { axis, step } => {
                    self.output_mapping[axis.index()].stepped(step)
                }
                AxisMapping::Broadcast => AxisMapping::Broadcast,
            })
            .collect();
        Self {
            input_shape: self.input_shape,
            input_padding: self.input_padding,
            input_offsets,
            output_mapping,
            output_shape: view.output_shape,
        }
    }

    pub(crate) fn transposed(&self) -> Self {
        assert_eq!(self.output_mapping.len(), 2);
        Self {
            input_shape: self.input_shape,
            input_padding: self.input_padding,
            input_offsets: self.input_offsets,
            output_mapping: self.output_mapping.iter().copied().rev().collect(),
            output_shape: self.output_shape.transposed(),
        }
    }

    pub(crate) fn broadcast(input_shape: &Shape, output_shape: &Shape) -> Self {
        assert!(input_shape.len() <= output_shape.len());
        let mut output_mapping = TinyVec::new();
        while output_mapping.len() + input_shape.len() < output_shape.len() {
            output_mapping.push(AxisMapping::Broadcast);
        }
        for (index, (&from, &to)) in input_shape
            .iter()
            .zip(output_shape.iter().skip(output_mapping.len()))
            .enumerate()
        {
            output_mapping.push(if from == to {
                AxisMapping::identity(Axis::from_index(index), from)
            } else {
                assert_eq!(from, 1);
                AxisMapping::Broadcast
            });
        }
        Self {
            input_shape: *input_shape,
            input_padding: iter::repeat(0).take(input_shape.len()).collect(),
            input_offsets: iter::repeat(0).take(input_shape.len()).collect(),
            output_mapping,
            output_shape: *output_shape,
        }
    }

    fn get_axis_step(&self, index: usize) -> Option<(Axis, isize)> {
        self.output_mapping
            .get(index)
            .and_then(|mapping| match mapping {
                AxisMapping::Source { axis, step } => Some((*axis, *step)),
                AxisMapping::Broadcast => None,
            })
    }

    pub(crate) fn load_column_major_hint(&self) -> bool {
        if let Some((axis0, step0)) = self.get_axis_step(0) {
            if let Some((axis1, step1)) = self.get_axis_step(1) {
                if axis0 == axis1 {
                    return step0.abs() < step1.abs();
                } else {
                    return axis0 > axis1;
                }
            }
        }
        false
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
