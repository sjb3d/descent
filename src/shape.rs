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

    #[must_use]
    pub(crate) fn prefix_ones_to_len(&self, len: usize) -> Self {
        let mut v = ShapeVec::new();
        while v.len() + self.len() < len {
            v.push(1);
        }
        v.extend_from_slice(self);
        Shape::new(v)
    }

    #[must_use]
    pub(crate) fn broadcast_with(&self, rhs: Shape) -> Self {
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

    pub(crate) fn batched_matmul(&self, rhs: Shape, output_mode: MatMulOutputMode) -> Self {
        let [b0, m, k0]: [usize; 3] = self.try_into().unwrap();
        let [b1, k1, n]: [usize; 3] = rhs.try_into().unwrap();
        assert_eq!(b0, b1);
        assert_eq!(k0, k1);
        let r = k0.div_round_up(MATMUL_MAX_K_SIZE);
        match output_mode {
            MatMulOutputMode::Batches => Shape::from([r, b0, m, n]),
            MatMulOutputMode::Rows => Shape::from([r, m, b0, n]),
        }
    }

    #[must_use]
    pub(crate) fn unpad(&self, axis: Axis, pad: usize) -> Self {
        let mut tmp = *self;
        tmp[axis] -= 2 * pad;
        tmp
    }

    #[must_use]
    pub(crate) fn pad(&self, axis: Axis, before: usize, after: usize) -> Self {
        let mut tmp = *self;
        tmp[axis] += before + after;
        tmp
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

    pub(crate) fn identity_view(&self) -> View {
        View::new(*self)
    }

    pub(crate) fn padded_view(&self, axis: Axis, before: usize, after: usize) -> View {
        View::new_padded(*self, axis, before, after)
    }

    pub(crate) fn identity_mapping(&self, axis: Axis) -> AxisMapping {
        AxisMapping::new(axis, self[axis])
    }

    pub(crate) fn iter_axes(&self) -> impl Iterator<Item = Axis> {
        (0..self.len()).map(Axis::from_index)
    }

    pub(crate) fn axis(&self, index: isize) -> Axis {
        // address from end if negative
        Axis::from_index((index as usize).wrapping_add(if index < 0 { self.0.len() } else { 0 }))
    }

    #[must_use]
    pub(crate) fn reduce(&self, axis: Axis) -> Self {
        let mut tmp = *self;
        tmp[axis] = 1;
        tmp
    }

    #[must_use]
    pub(crate) fn resize_axis(&self, axis: Axis, length: usize) -> Self {
        let mut tmp = *self;
        tmp[axis] = length;
        tmp
    }

    #[must_use]
    pub fn coord(&self, axis: Axis) -> Self {
        self.iter_axes()
            .map(|a| if a == axis { self[a] } else { 1 })
            .collect()
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

    #[must_use]
    pub(crate) fn insert_axis(&self, axis: Axis, len: usize) -> Self {
        let mut tmp = *self;
        tmp.0.insert(axis.index(), len);
        tmp
    }

    #[must_use]
    pub(crate) fn remove_axis(&self, axis: Axis) -> Self {
        let mut tmp = *self;
        tmp.0.remove(axis.index());
        tmp
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
impl iter::FromIterator<usize> for Shape {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
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

impl ops::Add for Shape {
    type Output = Shape;
    fn add(self, rhs: Self) -> Self::Output {
        let mut v = self.0;
        v.extend_from_slice(rhs.as_slice());
        Shape(v)
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
    fn new(axis: Axis, length: usize) -> Self {
        assert!(length > 0);
        if length > 1 {
            Self::Source { axis, step: 1 }
        } else {
            Self::Broadcast
        }
    }

    pub(crate) fn stepped(&self, step_multiplier: isize) -> Self {
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
    pub(crate) input_offsets: TinyVec<[isize; MAX_DIM]>,
    pub(crate) output_mapping: TinyVec<[AxisMapping; MAX_DIM]>,
    pub(crate) output_shape: Shape,
}

impl View {
    fn new(shape: Shape) -> Self {
        Self {
            input_shape: shape,
            input_offsets: iter::repeat(0).take(shape.len()).collect(),
            output_mapping: (0..shape.len())
                .map(|index| shape.identity_mapping(Axis::from_index(index)))
                .collect(),
            output_shape: shape,
        }
    }

    fn new_padded(shape: Shape, axis: Axis, before: usize, after: usize) -> Self {
        let mut tmp = View::new(shape);
        tmp.input_offsets[axis.index()] = -(before as isize);
        tmp.output_shape = tmp.output_shape.pad(axis, before, after);
        tmp
    }

    pub(crate) fn new_limited(
        shape: Shape,
        axis: Axis,
        range: impl ops::RangeBounds<usize>,
    ) -> Self {
        let start = match range.start_bound() {
            ops::Bound::Included(value) => *value,
            ops::Bound::Excluded(value) => *value + 1,
            ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            ops::Bound::Included(value) => *value + 1,
            ops::Bound::Excluded(value) => *value,
            ops::Bound::Unbounded => shape[axis],
        };
        let mut tmp = View::new(shape);
        tmp.input_offsets[axis.index()] = start as isize;
        tmp.output_mapping[axis.index()] = AxisMapping::new(axis, end - start);
        tmp.output_shape[axis] = end - start;
        tmp
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        let input_strides = self.input_shape.strides();
        let output_strides = self.output_shape.strides();
        self.input_shape.element_count() == self.output_shape.element_count()
            && self.input_offsets.iter().copied().all(|offset| offset == 0)
            && self
                .output_mapping
                .iter()
                .copied()
                .zip(output_strides.iter().copied())
                .all(|(output_mapping, output_stride)| match output_mapping {
                    AxisMapping::Broadcast => true,
                    AxisMapping::Source { axis, step } => {
                        (input_strides[axis.index()] as isize) * step == (output_stride as isize)
                    }
                })
    }

    pub(crate) fn try_from_reshape(input_shape: Shape, output_shape: Shape) -> Option<Self> {
        if input_shape == output_shape {
            return Some(input_shape.identity_view());
        }

        let mut output_mapping = TinyVec::new();
        for (input_axis, input_len) in input_shape.iter_axes().zip(input_shape.iter().copied()) {
            if input_len == 1 {
                continue;
            }
            loop {
                let output_len = *output_shape.get(output_mapping.len())?;
                if output_len == input_len {
                    output_mapping.push(AxisMapping::new(input_axis, input_len));
                    break;
                }
                if output_len != 1 {
                    return None;
                }
                output_mapping.push(AxisMapping::Broadcast);
            }
        }

        while output_mapping.len() < output_shape.len() {
            assert_eq!(output_shape[output_mapping.len()], 1);
            output_mapping.push(AxisMapping::Broadcast);
        }

        Some(Self {
            input_shape,
            input_offsets: iter::repeat(0).take(input_shape.len()).collect(),
            output_mapping,
            output_shape,
        })
    }

    fn input_axis_mapping_count(&self, input_axis: Axis) -> usize {
        self.output_mapping
            .iter()
            .filter(|mapping| match mapping {
                AxisMapping::Source { axis, .. } => *axis == input_axis,
                AxisMapping::Broadcast => false,
            })
            .count()
    }

    fn can_pad_output(&self, output_axis: Axis) -> bool {
        match self.output_mapping[output_axis.index()] {
            AxisMapping::Source {
                axis: input_axis,
                step,
            } => {
                // can only pad if this output fully covers a single input axis
                if self.input_axis_mapping_count(input_axis) == 1 {
                    let base = self.input_offsets[input_axis.index()];
                    let offset = ((self.output_shape[output_axis] - 1) as isize) * step;
                    let span_min = base + offset.min(0);
                    let span_max = base + offset.max(0);
                    let input_min = 0;
                    let input_max = (self.input_shape[input_axis] - 1) as isize;
                    span_min <= input_min && input_max <= span_max
                } else {
                    false
                }
            }
            AxisMapping::Broadcast => true, // padding a broadcast is still broadcast
        }
    }

    fn can_combine_with(&self, view: &View) -> bool {
        self.output_shape == view.input_shape
            && self
                .output_shape
                .iter_axes()
                .all(|axis| self.can_pad_output(axis) || !view.input_needs_clamp(axis))
    }

    fn can_reshape_to(&self, view: &View) -> bool {
        self.is_contiguous()
            && self.output_shape.element_count() == view.input_shape.element_count()
    }

    pub(crate) fn input_needs_clamp(&self, input_axis: Axis) -> bool {
        let mut offset_min = self.input_offsets[input_axis.index()];
        let mut offset_max = offset_min;
        for (mapping, len) in self
            .output_mapping
            .iter()
            .copied()
            .zip(self.output_shape.iter().copied())
        {
            match mapping {
                AxisMapping::Source { axis, step } => {
                    if axis == input_axis {
                        let offset = ((len - 1) as isize) * step;
                        offset_min += offset.min(0);
                        offset_max += offset.max(0);
                    }
                }
                AxisMapping::Broadcast => {}
            }
        }
        let input_min = 0;
        let input_max = (self.input_shape[input_axis] - 1) as isize;
        offset_min < input_min || input_max < offset_max
    }

    pub(crate) fn can_view_through(&self, view: &View, can_reshape: bool) -> bool {
        self.can_combine_with(view) || (can_reshape && self.can_reshape_to(view))
    }

    pub(crate) fn through(&self, view: &View, can_reshape: bool) -> Self {
        if !self.can_combine_with(view) {
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
            input_offsets,
            output_mapping,
            output_shape: view.output_shape,
        }
    }

    pub(crate) fn transposed(&self) -> Self {
        let len = self.output_mapping.len();
        assert!(len >= 2);
        let mut tmp = *self;
        tmp.output_mapping.swap(len - 2, len - 1);
        tmp.output_shape.0.swap(len - 2, len - 1);
        tmp
    }

    pub(crate) fn broadcast(input_shape: Shape, output_shape: Shape) -> Self {
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
                AxisMapping::new(Axis::from_index(index), from)
            } else {
                assert_eq!(from, 1);
                AxisMapping::Broadcast
            });
        }
        Self {
            input_shape,
            input_offsets: iter::repeat(0).take(input_shape.len()).collect(),
            output_mapping,
            output_shape,
        }
    }

    fn get_axis_step(&self, axis: isize) -> Option<(Axis, isize)> {
        let axis = self.output_shape.axis(axis);
        self.output_mapping
            .get(axis.index())
            .and_then(|mapping| match mapping {
                AxisMapping::Source { axis, step } => Some((*axis, *step)),
                AxisMapping::Broadcast => None,
            })
    }

    pub(crate) fn load_in_columns_hint(&self) -> bool {
        if let Some((outer_axis, outer_step)) = self.get_axis_step(-2) {
            if let Some((inner_axis, inner_step)) = self.get_axis_step(-1) {
                if outer_axis == inner_axis {
                    return outer_step.abs() < inner_step.abs();
                } else {
                    return outer_axis > inner_axis;
                }
            }
        }
        false
    }

    pub(crate) fn permute_axes(&self, perm: &[usize]) -> Self {
        let mut tmp = *self;
        tmp.output_mapping = perm
            .iter()
            .map(|&index| self.output_mapping[index])
            .collect();
        tmp.output_shape = perm.iter().map(|&index| self.output_shape[index]).collect();
        tmp
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_from_reshape() {
        assert!(View::try_from_reshape(Shape::from([8]), Shape::from([1, 1, 8])).is_some());
        assert!(View::try_from_reshape(Shape::from([8]), Shape::from([1, 8, 1])).is_some());
        assert!(View::try_from_reshape(Shape::from([8]), Shape::from([8, 1, 1])).is_some());

        assert!(View::try_from_reshape(Shape::from([1, 8, 1]), Shape::from([1, 1, 8])).is_some());
        assert!(View::try_from_reshape(Shape::from([1, 8, 1]), Shape::from([8, 1, 1])).is_some());

        assert!(View::try_from_reshape(Shape::from([8]), Shape::from([1, 9, 1])).is_none());
    }
}
