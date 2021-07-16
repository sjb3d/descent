use super::common::*;
use spark::vk;
use std::{
    collections::{HashMap, VecDeque},
    mem,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NameId {
    index: u32,
}

struct TimestampSet {
    context: SharedContext,
    query_pool: vk::QueryPool,
    timestamp_ids: Vec<NameId>,
}

impl TimestampSet {
    const MAX_QUERY_COUNT: usize = 128;

    fn new(context: &SharedContext) -> Self {
        let query_pool = {
            let create_info = vk::QueryPoolCreateInfo {
                query_type: vk::QueryType::TIMESTAMP,
                query_count: Self::MAX_QUERY_COUNT as u32,
                ..Default::default()
            };

            unsafe { context.device.create_query_pool(&create_info, None) }.unwrap()
        };
        Self {
            context: SharedContext::clone(context),
            query_pool,
            timestamp_ids: Vec::new(),
        }
    }

    fn write_timestamp(&mut self, cmd: vk::CommandBuffer, id: NameId) {
        if self.timestamp_ids.len() >= TimestampSet::MAX_QUERY_COUNT {
            return;
        }
        unsafe {
            self.context.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                self.timestamp_ids.len() as u32,
            )
        };
        self.timestamp_ids.push(id);
    }
}

struct TimestampAccumulator {
    context: SharedContext,
    timings_total: f32,
    timings_sum: Vec<(NameId, f32)>,
    timings_counter: u32,
    timestamp_valid_mask: u64,
    timestamp_period: f32,
}

impl TimestampAccumulator {
    fn new(context: &SharedContext) -> Self {
        Self {
            context: SharedContext::clone(context),
            timings_total: 0.0,
            timings_sum: Vec::new(),
            timings_counter: 0,
            timestamp_valid_mask: 1u64
                .checked_shl(context.queue_family_properties.timestamp_valid_bits)
                .unwrap_or(0)
                .wrapping_sub(1),
            timestamp_period: context.physical_device_properties.limits.timestamp_period
                / 1_000_000_000.0,
        }
    }

    fn accumulate_timings(&mut self, set: &mut TimestampSet) {
        if !set.timestamp_ids.is_empty() {
            let mut query_results = vec![0u64; set.timestamp_ids.len()];
            unsafe {
                self.context.device.get_query_pool_results(
                    set.query_pool,
                    0,
                    query_results.len() as u32,
                    &mut query_results,
                    mem::size_of::<u64>() as vk::DeviceSize,
                    vk::QueryResultFlags::N64 | vk::QueryResultFlags::WAIT,
                )
            }
            .unwrap();

            let query_deltas: Vec<u64> = (0..(set.timestamp_ids.len() - 1))
                .map(|i| {
                    let a = query_results[i];
                    let b = query_results[i + 1];
                    b.wrapping_sub(a) & self.timestamp_valid_mask
                })
                .collect();
            let query_times: Vec<f32> = query_deltas
                .iter()
                .copied()
                .map(|delta| (delta as f32) * self.timestamp_period)
                .collect();
            let total_time =
                (query_deltas.iter().copied().sum::<u64>() as f32) * self.timestamp_period;

            if self.timings_sum.len() == query_times.len()
                && self
                    .timings_sum
                    .iter()
                    .zip(set.timestamp_ids.iter().copied())
                    .all(|(a, b)| a.0 == b)
            {
                self.timings_total += total_time;
                for ((_, time_sum), time) in self.timings_sum.iter_mut().zip(query_times.iter()) {
                    *time_sum += time;
                }
                self.timings_counter += 1;
            } else {
                self.timings_total = total_time;
                self.timings_sum.clear();
                self.timings_sum.extend(
                    set.timestamp_ids
                        .iter()
                        .copied()
                        .zip(query_times.iter().copied()),
                );
                self.timings_counter = 1;
            }

            set.timestamp_ids.clear();
        }
    }

    fn print_timings(&self, names: &[String]) {
        if self.timings_counter != 0 {
            let norm = 1.0 / (self.timings_counter as f32);
            println!(
                "total ({} runs): {} us",
                self.timings_counter,
                norm * self.timings_total * 1_000_000.0
            );
            for (id, time) in self.timings_sum.iter() {
                let name = &names[id.index as usize];
                println!("{}: {} us", name, norm * time * 1_000_000.0);
            }
        }
    }

    fn reset_timings(&mut self) {
        self.timings_total = 0.0;
        self.timings_sum.clear();
        self.timings_counter = 0;
    }
}

pub(crate) struct TimestampSets {
    context: SharedContext,
    sets: VecDeque<Fenced<TimestampSet>>,
    name_ids: HashMap<String, NameId>,
    names: Vec<String>,
    accumulator: TimestampAccumulator,
}

impl TimestampSets {
    const COUNT: usize = 2;

    pub(crate) fn new(context: &SharedContext, fences: &FenceSet) -> Self {
        let mut sets = VecDeque::new();
        for _ in 0..Self::COUNT {
            sets.push_back(Fenced::new(TimestampSet::new(context), fences.old_id()));
        }
        Self {
            context: SharedContext::clone(context),
            sets,
            name_ids: HashMap::new(),
            names: Vec::new(),
            accumulator: TimestampAccumulator::new(context),
        }
    }

    fn name_id(&mut self, name: &str) -> NameId {
        if let Some(id) = self.name_ids.get(name) {
            *id
        } else {
            let id = NameId {
                index: self.names.len() as u32,
            };
            self.names.push(name.to_owned());
            self.name_ids.insert(name.to_owned(), id);
            id
        }
    }

    pub(crate) fn print_timings(&mut self, fences: &FenceSet) {
        // ensure all timings have been processed
        for set in self.sets.iter_mut() {
            self.accumulator
                .accumulate_timings(set.get_mut_when_signaled(fences));
        }
        self.accumulator.print_timings(&self.names);
        self.accumulator.reset_timings();
    }

    pub(crate) fn acquire(
        &mut self,
        cmd: vk::CommandBuffer,
        fences: &FenceSet,
    ) -> ScopedTimestampSet {
        let mut set = self.sets.pop_front().unwrap().take_when_signaled(fences);
        self.accumulator.accumulate_timings(&mut set);

        unsafe {
            self.context.device.cmd_reset_query_pool(
                cmd,
                set.query_pool,
                0,
                TimestampSet::MAX_QUERY_COUNT as u32,
            )
        };
        ScopedTimestampSet { set, owner: self }
    }
}

impl Drop for TimestampSets {
    fn drop(&mut self) {
        let device = &self.context.device;
        for set in self.sets.iter() {
            unsafe {
                let set = set.get_unchecked();
                device.destroy_query_pool(Some(set.query_pool), None);
            }
        }
    }
}

pub(crate) struct ScopedTimestampSet<'a> {
    set: TimestampSet,
    owner: &'a mut TimestampSets,
}

impl<'a> ScopedTimestampSet<'a> {
    pub(crate) fn write_timestamp(&mut self, cmd: vk::CommandBuffer, name: &str) {
        self.set.write_timestamp(cmd, self.owner.name_id(name))
    }

    pub(crate) fn end(&mut self, cmd: vk::CommandBuffer) {
        if let Some(id) = self.set.timestamp_ids.last().copied() {
            self.set.write_timestamp(cmd, id);
        }
    }

    pub(crate) fn recycle(self, fence: FenceId) {
        self.owner.sets.push_back(Fenced::new(self.set, fence));
    }
}
