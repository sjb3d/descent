use spark::{vk, Builder, Device, Instance, Loader};
use std::rc::Rc;
use std::{ffi::CStr, slice};

trait PhysicalDeviceMemoryPropertiesExt {
    fn types(&self) -> &[vk::MemoryType];
    fn heaps(&self) -> &[vk::MemoryHeap];
}

impl PhysicalDeviceMemoryPropertiesExt for vk::PhysicalDeviceMemoryProperties {
    fn types(&self) -> &[vk::MemoryType] {
        &self.memory_types[..self.memory_type_count as usize]
    }
    fn heaps(&self) -> &[vk::MemoryHeap] {
        &self.memory_heaps[..self.memory_heap_count as usize]
    }
}

pub struct Context {
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub queue_family_properties: vk::QueueFamilyProperties,
    pub queue: vk::Queue,
    pub device: Device,
}

pub type SharedContext = Rc<Context>;

impl Context {
    pub fn new() -> SharedContext {
        let version = vk::Version::default();
        let instance = {
            let loader = Loader::new().unwrap();

            let app_info = vk::ApplicationInfo::builder()
                .p_application_name(Some(CStr::from_bytes_with_nul(b"caldera\0").unwrap()))
                .api_version(version);

            let instance_create_info =
                vk::InstanceCreateInfo::builder().p_application_info(Some(&app_info));
            unsafe { loader.create_instance(&instance_create_info, None) }.unwrap()
        };

        let physical_device = {
            let physical_devices = unsafe { instance.enumerate_physical_devices_to_vec() }.unwrap();
            for physical_device in &physical_devices {
                let props = unsafe { instance.get_physical_device_properties(*physical_device) };
                println!("physical device ({}): {:?}", props.device_type, unsafe {
                    CStr::from_ptr(props.device_name.as_ptr())
                });
            }
            physical_devices[0]
        };
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        for (i, mt) in physical_device_memory_properties.types().iter().enumerate() {
            println!(
                "memory type {}: {}, heap {}",
                i, mt.property_flags, mt.heap_index
            );
        }
        for (i, mh) in physical_device_memory_properties.heaps().iter().enumerate() {
            println!("heap {}: {} bytes {}", i, mh.size, mh.flags);
        }

        let (queue_family_index, queue_family_properties) = {
            let queue_flags = vk::QueueFlags::COMPUTE;

            unsafe { instance.get_physical_device_queue_family_properties_to_vec(physical_device) }
                .iter()
                .enumerate()
                .filter_map(|(index, info)| {
                    if info.queue_flags.contains(queue_flags) {
                        Some((index as u32, *info))
                    } else {
                        None
                    }
                })
                .next()
                .unwrap()
        };

        let device = {
            let queue_priorities = [1.0];
            let device_queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .p_queue_priorities(&queue_priorities);

            let device_create_info = vk::DeviceCreateInfo::builder()
                .p_queue_create_infos(slice::from_ref(&device_queue_create_info));

            unsafe { instance.create_device(physical_device, &device_create_info, None, version) }
                .unwrap()
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        SharedContext::new(Self {
            instance,
            physical_device,
            physical_device_properties,
            physical_device_memory_properties,
            queue_family_index,
            queue_family_properties,
            queue,
            device,
        })
    }

    pub fn get_memory_type_index(
        &self,
        type_filter: u32,
        property_flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for (i, mt) in self
            .physical_device_memory_properties
            .types()
            .iter()
            .enumerate()
        {
            let i = i as u32;
            if (type_filter & (1 << i)) != 0 && mt.property_flags.contains(property_flags) {
                return Some(i);
            }
        }
        None
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
