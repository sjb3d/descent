use spark::{vk, Builder, Device, DeviceExtensions, Instance, InstanceExtensions, Loader};
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

pub(crate) struct Context {
    pub(crate) instance: Instance,
    pub(crate) _physical_device: vk::PhysicalDevice,
    pub(crate) physical_device_properties: vk::PhysicalDeviceProperties,
    pub(crate) physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) queue_family_index: u32,
    pub(crate) queue_family_properties: vk::QueueFamilyProperties,
    pub(crate) queue: vk::Queue,
    pub(crate) device: Device,
    pub(crate) has_shader_atomic_float: bool,
}

pub(crate) type SharedContext = Rc<Context>;

impl Context {
    pub(crate) fn new() -> SharedContext {
        let version = vk::Version::default();
        let instance = {
            let loader = Loader::new().unwrap();

            let available_extensions = {
                let extension_properties =
                    unsafe { loader.enumerate_instance_extension_properties_to_vec(None) }.unwrap();
                InstanceExtensions::from_properties(version, &extension_properties)
            };

            let mut extensions = InstanceExtensions::new(version);
            if available_extensions.supports_ext_debug_utils() {
                extensions.enable_ext_debug_utils();
            }
            if available_extensions.supports_ext_shader_atomic_float() {
                extensions.enable_ext_shader_atomic_float();
            }
            let extension_names = extensions.to_name_vec();

            let app_info = vk::ApplicationInfo::builder()
                .p_application_name(Some(CStr::from_bytes_with_nul(b"caldera\0").unwrap()))
                .api_version(version);

            let extension_name_ptrs: Vec<_> = extension_names.iter().map(|s| s.as_ptr()).collect();
            let instance_create_info = vk::InstanceCreateInfo::builder()
                .p_application_info(Some(&app_info))
                .pp_enabled_extension_names(&extension_name_ptrs);
            unsafe { loader.create_instance(&instance_create_info, None) }.unwrap()
        };

        let physical_device = {
            let physical_devices = unsafe { instance.enumerate_physical_devices_to_vec() }.unwrap();
            for (i, physical_device) in physical_devices.iter().enumerate() {
                let props = unsafe { instance.get_physical_device_properties(*physical_device) };
                println!(
                    "physical device {}: {:?} ({})",
                    i,
                    unsafe { CStr::from_ptr(props.device_name.as_ptr()) },
                    props.device_type
                );
            }
            physical_devices[0]
        };
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

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

        let mut has_shader_atomic_float = false;
        let device = {
            let queue_priorities = [1.0];
            let device_queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .p_queue_priorities(&queue_priorities);

            let available_extensions = {
                let extension_properties = unsafe {
                    instance.enumerate_device_extension_properties_to_vec(physical_device, None)
                }
                .unwrap();
                DeviceExtensions::from_properties(version, &extension_properties)
            };

            let mut extensions = DeviceExtensions::new(version);
            let mut shader_atomic_float_features =
                vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
            if available_extensions.supports_ext_shader_atomic_float() {
                extensions.enable_ext_shader_atomic_float();
                shader_atomic_float_features.shader_buffer_float32_atomic_add = vk::TRUE;
                has_shader_atomic_float = true;
            }
            let extension_names = extensions.to_name_vec();

            let extension_name_ptrs: Vec<_> = extension_names.iter().map(|s| s.as_ptr()).collect();
            let device_create_info = vk::DeviceCreateInfo::builder()
                .p_queue_create_infos(slice::from_ref(&device_queue_create_info))
                .pp_enabled_extension_names(&extension_name_ptrs)
                .insert_next(&mut shader_atomic_float_features);

            unsafe { instance.create_device(physical_device, &device_create_info, None, version) }
                .unwrap()
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        SharedContext::new(Self {
            instance,
            _physical_device: physical_device,
            physical_device_properties,
            physical_device_memory_properties,
            queue_family_index,
            queue_family_properties,
            queue,
            device,
            has_shader_atomic_float,
        })
    }

    pub(crate) fn get_memory_type_index(
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
