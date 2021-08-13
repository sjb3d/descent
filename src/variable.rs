use crate::{common::*, device::common::*};
use slotmap::SlotMap;
use std::{cell::RefCell, rc::Rc};

slotmap::new_key_type! {
    pub(crate) struct VariableId;
}

#[derive(Clone, Copy, Debug)]
pub enum Initializer {
    Zero,
    RandNormal(f32),
    RandUniform(f32),
}

impl Initializer {
    pub fn for_relu(fan_in: usize) -> Self {
        let scale = (2.0 / (fan_in as f32)).sqrt();
        Self::RandNormal(scale)
    }

    pub fn for_siren(fan_in: usize) -> Self {
        let scale = (6.0 / (fan_in as f32)).sqrt();
        Self::RandUniform(scale)
    }
}

pub(crate) struct VariableStorage {
    pub(crate) shape: Shape,
    pub(crate) name: String,
    pub(crate) buffer_id: Option<BufferId>,
    pub(crate) reset_to: Option<Initializer>,
}

pub(crate) type SharedVariables = Rc<RefCell<SlotMap<VariableId, VariableStorage>>>;

#[derive(Clone)]
pub struct Variable {
    id: VariableId,
    owner: SharedVariables,
}

impl Variable {
    pub(crate) fn new(variable_id: VariableId, owner: &SharedVariables) -> Self {
        Self {
            id: variable_id,
            owner: SharedVariables::clone(owner),
        }
    }

    pub(crate) fn checked_id(&self, owner: &SharedVariables) -> VariableId {
        if !SharedVariables::ptr_eq(&self.owner, owner) {
            panic!("variable does not come from the same environment");
        }
        self.id
    }

    pub fn shape(&self) -> Shape {
        self.owner.borrow().get(self.id).unwrap().shape
    }

    pub fn name(&self) -> String {
        self.owner.borrow().get(self.id).unwrap().name.clone()
    }

    pub fn reset_to(&self) -> Option<Initializer> {
        self.owner.borrow().get(self.id).unwrap().reset_to
    }

    pub fn is_trainable(&self) -> bool {
        self.owner.borrow().get(self.id).unwrap().reset_to.is_some()
    }
}
