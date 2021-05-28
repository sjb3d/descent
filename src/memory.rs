use crate::{context::*, prelude::*};
use slotmap::SlotMap;
use std::{ops::Range, rc::Rc};

slotmap::new_key_type! {
    pub(crate) struct HeapBlockId;
}

#[derive(Debug, Clone, Copy)]
struct HeapBlockIdLink {
    prev: HeapBlockId,
    next: HeapBlockId,
}

impl HeapBlockIdLink {
    fn new(id: HeapBlockId) -> Self {
        Self { prev: id, next: id }
    }

    fn is_self(&self, id: HeapBlockId) -> bool {
        self.prev == id && self.next == id
    }
}

#[derive(Debug, Clone, Copy)]
struct HeapRange {
    begin: usize,
    end: usize,
}

impl HeapRange {
    fn size(&self) -> usize {
        self.end - self.begin
    }

    fn truncate(&mut self, new_size: usize) -> HeapRange {
        assert!(new_size > 0);
        let begin = self.begin + new_size;
        let end = self.end;
        assert!(begin < end);
        self.end = begin;
        HeapRange { begin, end }
    }
}

#[derive(Debug, Clone, Copy)]
struct HeapBlock {
    range: HeapRange,
    all_link: HeapBlockIdLink,
    free_link: HeapBlockIdLink,
}

impl HeapBlock {
    fn new(id: HeapBlockId, begin: usize, end: usize) -> Self {
        Self {
            range: HeapRange { begin, end },
            all_link: HeapBlockIdLink::new(id),
            free_link: HeapBlockIdLink::new(id),
        }
    }
}

type HeapBlockSlotMap = SlotMap<HeapBlockId, HeapBlock>;

pub(crate) struct HeapAllocator {
    blocks: HeapBlockSlotMap,
    free_list_sentinels: Vec<HeapBlockId>,
}

impl HeapAllocator {
    fn free_list_index(size: usize) -> usize {
        (0usize.leading_zeros() - size.leading_zeros()) as usize
    }

    pub(crate) fn new(size: usize) -> Self {
        let max_free_list_index = Self::free_list_index(size);

        let mut blocks = SlotMap::with_key();
        let mut free_list_sentinels = Vec::new();
        for i in 0..=max_free_list_index {
            free_list_sentinels.push(blocks.insert_with_key(|key| HeapBlock::new(key, 0, 0)));
        }

        let id = blocks.insert_with_key(|key| HeapBlock::new(key, 0, size));
        Self::link_free_block(&mut blocks, &free_list_sentinels, id);

        Self {
            blocks,
            free_list_sentinels,
        }
    }

    fn free_links(
        blocks: &mut HeapBlockSlotMap,
        left_id: HeapBlockId,
        middle_id: HeapBlockId,
        right_id: HeapBlockId,
    ) -> (&mut HeapBlockId, &mut HeapBlockIdLink, &mut HeapBlockId) {
        if left_id == right_id {
            let [other, middle] = blocks.get_disjoint_mut([left_id, middle_id]).unwrap();
            (
                &mut other.free_link.next,
                &mut middle.free_link,
                &mut other.free_link.prev,
            )
        } else {
            let [left, middle, right] = blocks
                .get_disjoint_mut([left_id, middle_id, right_id])
                .unwrap();
            (
                &mut left.free_link.next,
                &mut middle.free_link,
                &mut right.free_link.prev,
            )
        }
    }

    fn link_free_block(
        blocks: &mut HeapBlockSlotMap,
        free_list_sentinels: &[HeapBlockId],
        alloc_id: HeapBlockId,
    ) {
        let free_list_index = Self::free_list_index(blocks[alloc_id].range.size());
        let left_id = free_list_sentinels[free_list_index];
        let right_id = blocks[left_id].free_link.next;
        let (left_next, link, right_prev) = Self::free_links(blocks, left_id, alloc_id, right_id);
        assert!(link.is_self(alloc_id));
        assert_eq!(*right_prev, left_id);
        *left_next = alloc_id;
        link.prev = left_id;
        link.next = right_id;
        *right_prev = alloc_id;
    }

    fn unlink_free_block(blocks: &mut HeapBlockSlotMap, free_id: HeapBlockId) {
        let HeapBlockIdLink {
            prev: left_id,
            next: right_id,
        } = blocks[free_id].free_link;
        let (left_next, link, right_prev) = Self::free_links(blocks, left_id, free_id, right_id);

        assert_eq!(*left_next, free_id);
        assert_eq!(*right_prev, free_id);
        *left_next = right_id;
        link.prev = free_id;
        link.next = free_id;
        *right_prev = left_id;
    }

    fn all_links(
        blocks: &mut HeapBlockSlotMap,
        left_id: HeapBlockId,
        middle_id: HeapBlockId,
        right_id: HeapBlockId,
    ) -> (
        &mut HeapBlockId,
        &mut HeapRange,
        &mut HeapBlock,
        &mut HeapBlockId,
    ) {
        if left_id == right_id {
            let [other, middle] = blocks.get_disjoint_mut([left_id, middle_id]).unwrap();
            (
                &mut other.all_link.next,
                &mut other.range,
                middle,
                &mut other.all_link.prev,
            )
        } else {
            let [left, middle, right] = blocks
                .get_disjoint_mut([left_id, middle_id, right_id])
                .unwrap();
            (
                &mut left.all_link.next,
                &mut left.range,
                middle,
                &mut right.all_link.prev,
            )
        }
    }

    fn truncate_block(
        blocks: &mut HeapBlockSlotMap,
        orig_id: HeapBlockId,
        new_size: usize,
    ) -> HeapBlockId {
        let new_id = blocks.insert_with_key(|key| HeapBlock::new(key, 0, 0));
        let next_id = blocks[orig_id].all_link.next;
        let (left_next, left_range, new_block, right_prev) =
            Self::all_links(blocks, orig_id, new_id, next_id);

        assert_eq!(*right_prev, orig_id);
        *left_next = new_id;
        new_block.all_link.prev = orig_id;
        new_block.all_link.next = next_id;
        *right_prev = new_id;

        new_block.range = left_range.truncate(new_size);

        new_id
    }

    fn append_block(blocks: &mut HeapBlockSlotMap, orig_id: HeapBlockId, append_id: HeapBlockId) {
        let next_id = blocks[append_id].all_link.next;
        let (left_next, left_range, append_block, right_prev) =
            Self::all_links(blocks, orig_id, append_id, next_id);

        assert!(append_block.free_link.is_self(append_id));
        *left_next = next_id;
        *right_prev = orig_id;

        assert_eq!(left_range.end, append_block.range.begin);
        left_range.end = append_block.range.end;

        blocks.remove(append_id).unwrap();
    }

    fn print_state(&self) {
        for (index, sentinel_id) in self.free_list_sentinels.iter().cloned().enumerate() {
            println!("free list {}:", index);
            let mut block_id = self.blocks[sentinel_id].free_link.next;
            while block_id != sentinel_id {
                let block = &self.blocks[block_id];
                println!("{:?} = {:?}", block_id, block);
                block_id = block.free_link.next;
            }
        }
        println!("full list:");
        if let Some((first_block_id, _)) = self
            .blocks
            .iter()
            .filter(|(_, block)| block.range.end != 0)
            .next()
        {
            let mut block_id = first_block_id;
            loop {
                let block = &self.blocks[block_id];
                println!("{:?} = {:?}", block_id, block);
                block_id = block.all_link.next;
                if block_id == first_block_id {
                    break;
                }
            }
        }
    }

    pub(crate) fn alloc(&mut self, size: usize, align: usize) -> Option<(HeapBlockId, usize)> {
        let blocks = &mut self.blocks;
        let free_list_sentinels = &self.free_list_sentinels;

        let align_mask = align - 1;
        let start_free_list_index = Self::free_list_index(size);
        for sentinel_id in free_list_sentinels[start_free_list_index..].iter().cloned() {
            let mut block_id = blocks[sentinel_id].free_link.next;
            while block_id != sentinel_id {
                let block_range = blocks[block_id].range;
                let aligned_begin = (block_range.begin + align_mask) & !align_mask;
                let aligned_end = aligned_begin + size;
                if aligned_end <= block_range.end {
                    Self::unlink_free_block(blocks, block_id);
                    if aligned_begin != block_range.begin {
                        let aligned_id = Self::truncate_block(
                            blocks,
                            block_id,
                            aligned_begin - block_range.begin,
                        );
                        Self::link_free_block(blocks, free_list_sentinels, block_id);
                        block_id = aligned_id;
                    }
                    if aligned_end != block_range.end {
                        let unused_id = Self::truncate_block(blocks, block_id, size);
                        Self::link_free_block(blocks, free_list_sentinels, unused_id);
                    }
                    return Some((block_id, aligned_begin));
                }
                block_id = blocks[block_id].free_link.next;
            }
        }
        None
    }

    pub(crate) fn free(&mut self, block_id: HeapBlockId) {
        let blocks = &mut self.blocks;
        let free_list_sentinels = &self.free_list_sentinels;

        let block = &blocks[block_id];
        assert_eq!(block.free_link.prev, block_id);
        assert_eq!(block.free_link.next, block_id);
        let next_id = block.all_link.next;
        let next = &blocks[next_id];
        if !next.free_link.is_self(next_id) && block.range.end == next.range.begin {
            Self::unlink_free_block(blocks, next_id);
            Self::append_block(blocks, block_id, next_id);
        }

        let block = &blocks[block_id];
        let prev_id = block.all_link.prev;
        let prev = &blocks[prev_id];
        if !prev.free_link.is_self(prev_id) && prev.range.end == block.range.begin {
            Self::unlink_free_block(blocks, prev_id);
            Self::append_block(blocks, prev_id, block_id);
            Self::link_free_block(blocks, free_list_sentinels, prev_id);
        } else {
            Self::link_free_block(blocks, free_list_sentinels, block_id);
        }
    }
}

pub fn memory_test() {
    let mut heap = HeapAllocator::new(1000);

    let (ai, ao) = heap.alloc(1000, 4).unwrap();
    println!("alloc(1000) = {}", ao);
    heap.print_state();
    heap.free(ai);
    println!("free(1000)");
    heap.print_state();

    let (ai, ao) = heap.alloc(500, 4).unwrap();
    println!("alloc(500) = {}", ao);
    heap.print_state();
    let (bi, bo) = heap.alloc(500, 4).unwrap();
    println!("alloc(500) = {}", bo);
    heap.print_state();
    heap.free(ai);
    println!("free(500)");
    heap.print_state();
    let (ci, co) = heap.alloc(250, 2).unwrap();
    println!("alloc(250) = {}", co);
    heap.print_state();
    let (di, dof) = heap.alloc(250, 2).unwrap();
    println!("alloc(250) = {}", dof);
    heap.print_state();

    println!("free all");
    heap.free(bi);
    heap.free(ci);
    heap.free(di);
    heap.print_state();
}
