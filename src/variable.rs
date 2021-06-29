use crate::{common::*, device::common::*};
use slotmap::SlotMap;
use std::{cell::RefCell, rc::Rc};

slotmap::new_key_type! {
    pub(crate) struct VariableId;
}

pub(crate) struct VariableStorage {
    pub(crate) shape: Shape,
    pub(crate) name: String,
    pub(crate) buffer_id: Option<BufferId>,
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
        self.owner.borrow().get(self.id).unwrap().shape.clone()
    }

    pub fn name(&self) -> String {
        self.owner.borrow().get(self.id).unwrap().name.clone()
    }
}
