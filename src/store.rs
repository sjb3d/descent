use std::{marker::PhantomData, ops};

pub trait Idx: Copy {
    fn new(index: usize) -> Self;

    fn index(self) -> usize;
}

pub struct Store<I: Idx, T> {
    slots: Vec<Option<T>>,
    _marker: PhantomData<*mut I>,
}

impl<I: Idx, T> Store<I, T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn add(&mut self, item: T) -> I {
        let index = self.slots.len();
        self.slots.push(Some(item));
        I::new(index)
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (I, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.as_ref().map(|item| (I::new(index), item)))
    }

    pub fn retain(&mut self, mut f: impl FnMut(I) -> bool) {
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.is_some() && !f(I::new(index)) {
                slot.take();
            }
        }
    }
}

impl<I: Idx, T: Clone> Clone for Store<I, T> {
    fn clone(&self) -> Self {
        Self {
            slots: self.slots.clone(),
            _marker: PhantomData,
        }
    }
}

impl<I: Idx, T> ops::Index<I> for Store<I, T> {
    type Output = T;
    fn index(&self, id: I) -> &Self::Output {
        self.slots.index(id.index()).as_ref().unwrap()
    }
}
impl<I: Idx, T> ops::IndexMut<I> for Store<I, T> {
    fn index_mut(&mut self, id: I) -> &mut Self::Output {
        self.slots.index_mut(id.index()).as_mut().unwrap()
    }
}
