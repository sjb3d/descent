use std::{convert::TryInto, fmt, iter, mem, ops};
use tinyvec::ArrayVec as TinyVec;

pub(crate) const MAX_DIM: usize = 6;
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

    pub(crate) fn inner(&self) -> Self {
        Self(self.0 + 1)
    }
}

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
        let [m, k0]: [usize; 2] = self.as_slice().try_into().unwrap();
        let [k1, n]: [usize; 2] = rhs.as_slice().try_into().unwrap();
        assert_eq!(k0, k1);
        Shape::from([m, n])
    }

    pub(crate) fn image_to_windows(&self, filter_w: usize, filter_h: usize, pad: usize) -> Self {
        assert!(self.0.len() >= 3);
        let (prefix, suffix) = self.rsplit_at(3);
        let [in_h, in_w, in_nc]: [usize; 3] = suffix.try_into().unwrap();
        let out_w = 1 + in_w + 2 * pad - filter_w;
        let out_h = 1 + in_h + 2 * pad - filter_h;
        let mut v = TinyVec::new();
        v.extend_from_slice(prefix);
        v.extend_from_slice(&[out_h, out_w, filter_h, filter_w, in_nc]);
        Shape::new(v)
    }

    pub(crate) fn windows_to_image(&self, pad: usize) -> Self {
        assert!(self.0.len() >= 5);
        let (prefix, suffix) = self.rsplit_at(5);
        let [out_h, out_w, filter_h, filter_w, in_nc]: [usize; 5] = suffix.try_into().unwrap();
        let in_w = out_w + filter_w - 1 - 2 * pad;
        let in_h = out_h + filter_h - 1 - 2 * pad;
        let mut v = TinyVec::new();
        v.extend_from_slice(prefix);
        v.extend_from_slice(&[in_h, in_w, in_nc]);
        Shape::new(v)
    }

    pub(crate) fn pool(&self, axis: isize, size: usize) -> Self {
        let axis = self.axis(axis);
        let (prefix, suffix) = self.0.split_at(axis.index());
        let (src_size, suffix) = suffix.split_first().unwrap();
        let src_size = *src_size;
        assert_eq!(src_size % size, 0, "pooling must exactly divide the axis");
        let mut v = ShapeVec::new();
        v.extend_from_slice(prefix);
        v.push(src_size / size);
        v.push(size);
        v.extend_from_slice(suffix);
        Shape::new(v)
    }

    pub(crate) fn transposed(&self) -> Self {
        let [a, b]: [usize; 2] = self.as_slice().try_into().unwrap();
        Shape::from([b, a])
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
            let mut v = self.0;
            v[index] = 1;
            Shape::new(v)
        }
    }

    pub(crate) fn one_hot(&self, count: usize) -> Self {
        // expand last axis (innermost dimension) from 1 to n
        let (last, prefix) = self.0.split_last().unwrap();
        assert_eq!(*last, 1);
        let mut v = TinyVec::new();
        v.extend_from_slice(prefix);
        v.push(count);
        Shape::new(v)
    }

    fn strides(&self) -> TinyVec<[isize; MAX_DIM]> {
        let mut stride = 1;
        let v: TinyVec<[isize; MAX_DIM]> = self
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

impl ops::Index<Axis> for Shape {
    type Output = usize;
    fn index(&self, axis: Axis) -> &Self::Output {
        self.as_slice().index(axis.index())
    }
}
impl ops::IndexMut<Axis> for Shape {
    fn index_mut(&mut self, axis: Axis) -> &mut Self::Output {
        self.as_mut_slice().index_mut(axis.index())
    }
}
impl ops::Index<isize> for Shape {
    type Output = usize;
    fn index(&self, index: isize) -> &Self::Output {
        self.index(self.axis(index))
    }
}
impl ops::IndexMut<isize> for Shape {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        self.index_mut(self.axis(index))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AxisMapping {
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
    input_offsets: TinyVec<[isize; MAX_DIM]>,
    output_mapping: TinyVec<[AxisMapping; MAX_DIM]>,
    pub(crate) output_shape: Shape,
}

impl View {
    fn identity(shape: &Shape) -> Self {
        Self {
            input_shape: *shape,
            input_offsets: iter::repeat(0).take(shape.len()).collect(),
            output_mapping: shape
                .iter()
                .copied()
                .enumerate()
                .map(|(index, shape)| AxisMapping::identity(Axis::from_index(index), shape))
                .collect(),
            output_shape: *shape,
        }
    }

    pub(crate) fn is_identity(&self) -> bool {
        Self::identity(&self.output_shape).eq(self)
    }

    pub(crate) fn can_view_through(&self, view: &View) -> bool {
        self.is_identity() || self.output_shape == view.input_shape
    }

    pub(crate) fn through(&self, view: &View) -> Self {
        if self.is_identity() {
            return *view;
        }

        assert_eq!(&self.output_shape, &view.input_shape);
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
        assert_eq!(self.output_mapping.len(), 2);
        Self {
            input_shape: self.input_shape,
            input_offsets: self.input_offsets,
            output_mapping: self.output_mapping.iter().copied().rev().collect(),
            output_shape: self.output_shape.transposed(),
        }
    }

    pub(crate) fn broadcast(input_shape: &Shape, output_shape: &Shape) -> Self {
        assert!(input_shape.len() <= output_shape.len());
        let input_offsets = iter::repeat(0).take(input_shape.len()).collect();
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
                AxisMapping::Broadcast
            });
        }
        Self {
            input_shape: *input_shape,
            input_offsets,
            output_mapping,
            output_shape: *output_shape,
        }
    }

    pub(crate) fn indexer(&self) -> ViewIndexer {
        ViewIndexer::new(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ViewIndexer {
    pub(crate) scales: TinyVec<[isize; MAX_DIM]>,
    pub(crate) offset: isize,
}

impl ViewIndexer {
    fn new(view: &View) -> Self {
        let input_strides = view.input_shape.strides();

        let scales = view
            .output_mapping
            .iter()
            .map(|mapping| match mapping {
                AxisMapping::Source { axis, step } => input_strides[axis.index()] * step,
                AxisMapping::Broadcast => 0,
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
