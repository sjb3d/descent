use crate::{common::*, device::common::*};
use slotmap::SlotMap;
use std::{cell::RefCell, rc::Rc};

slotmap::new_key_type! {
    pub(crate) struct ParameterId;
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

    pub fn for_siren(fan_in: usize, is_first_layer: bool) -> Self {
        let scale = (6.0 / (fan_in as f32)).sqrt() * if is_first_layer { 30.0 } else { 1.0 };
        Self::RandUniform(scale)
    }
}

pub(crate) struct ParameterStorage {
    pub(crate) shape: Shape,
    pub(crate) name: String,
    pub(crate) buffer_id: Option<BufferId>,
    pub(crate) reset_to: Option<Initializer>,
}

pub(crate) type SharedParameters = Rc<RefCell<SlotMap<ParameterId, ParameterStorage>>>;

#[derive(Clone)]
pub struct Parameter {
    id: ParameterId,
    owner: SharedParameters,
}

impl Parameter {
    pub(crate) fn new(parameter_id: ParameterId, owner: &SharedParameters) -> Self {
        Self {
            id: parameter_id,
            owner: SharedParameters::clone(owner),
        }
    }

    pub(crate) fn checked_id(&self, owner: &SharedParameters) -> ParameterId {
        if !SharedParameters::ptr_eq(&self.owner, owner) {
            panic!("parameter does not come from the same environment");
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
