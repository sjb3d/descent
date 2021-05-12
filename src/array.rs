use arrayvec::ArrayVec;
use std::fmt;

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

    fn inner(&self) -> Option<Self> {
        self.0
            .split_first()
            .map(|(_, remain)| remain)
            .filter(|remain| !remain.is_empty())
            .map(|remain| Self(remain.iter().cloned().collect()))
    }
}

impl<const N: usize> Into<Size> for [usize; N] {
    fn into(self) -> Size {
        Size(self.iter().cloned().collect())
    }
}

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
        Self {
            elements: vec![0.0; size.elements()],
            size,
        }
    }

    pub fn size(&self) -> &Size {
        &self.size
    }

    fn view(&self) -> ArrayView {
        ArrayView {
            elements: &self.elements,
            size: self.size.clone(),
        }
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.view().fmt(fmt)
    }
}

struct ArrayView<'a> {
    elements: &'a [f32],
    size: Size,
}

impl<'a> fmt::Debug for ArrayView<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(inner) = self.size.inner() {
            fmt.debug_list()
                .entries(self.elements.chunks(inner.elements()).map(|s| ArrayView {
                    elements: s,
                    size: inner.clone(),
                }))
                .finish()
        } else {
            fmt.debug_list().entries(self.elements.iter()).finish()
        }
    }
}
