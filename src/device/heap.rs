use slotmap::{Key, SlotMap};
use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
struct BlockListNode<K: Key> {
    prev_id: K,
    next_id: K,
}

impl<K: Key> BlockListNode<K> {
    fn new(id: K) -> Self {
        Self {
            prev_id: id,
            next_id: id,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Range {
    begin: usize,
    end: usize,
}

impl Range {
    fn new(size: usize) -> Self {
        Self {
            begin: 0,
            end: size,
        }
    }

    fn size(&self) -> usize {
        self.end - self.begin
    }

    fn truncate(&mut self, new_size: usize) -> Range {
        assert!(new_size > 0);
        let begin = self.begin + new_size;
        let end = self.end;
        assert!(begin < end);
        self.end = begin;
        Range { begin, end }
    }

    fn append(&mut self, other: Range) {
        assert_eq!(self.end, other.begin);
        self.end = other.end;
    }
}

pub trait Tag: Debug + Clone + Copy + PartialEq + Eq {}

#[derive(Debug, Clone, Copy)]
struct Block<K: Key, T: Tag> {
    tag: T,
    range: Range,
    tag_node: BlockListNode<K>,
    free_node: Option<BlockListNode<K>>,
}

impl<K: Key, T: Tag> Block<K, T> {
    fn new(id: K, tag: T, range: Range) -> Self {
        Self {
            tag,
            range,
            tag_node: BlockListNode::new(id),
            free_node: None,
        }
    }

    fn can_append(&self, other: &Block<K, T>) -> bool {
        self.tag == other.tag && self.range.end == other.range.begin
    }

    fn tag(&self) -> (T, usize) {
        (self.tag, self.range.begin)
    }
}

type BlockSlotMap<K, T> = SlotMap<K, Block<K, T>>;

#[derive(Debug)]
pub struct Heap<K: Key, T: Tag> {
    blocks: BlockSlotMap<K, T>,
    free_lists: Vec<Option<K>>,
}

impl<K: Key, T: Tag> Default for Heap<K, T> {
    fn default() -> Self {
        Self {
            blocks: BlockSlotMap::with_key(),
            free_lists: Vec::new(),
        }
    }
}

impl<K: Key, T: Tag> Heap<K, T> {
    fn free_list_index(size: usize) -> usize {
        (0usize.leading_zeros() - size.leading_zeros()) as usize
    }

    pub fn extend_with(&mut self, tag: T, size: usize) {
        let free_list_index = Self::free_list_index(size);

        while free_list_index >= self.free_lists.len() {
            self.free_lists.push(None);
        }

        let id = self
            .blocks
            .insert_with_key(|key| Block::new(key, tag, Range::new(size)));
        Self::register_free_block(&mut self.blocks, self.free_lists.as_mut_slice(), id);
    }

    fn register_free_block(
        blocks: &mut BlockSlotMap<K, T>,
        free_lists: &mut [Option<K>],
        alloc_id: K,
    ) {
        let size = {
            let block = &blocks[alloc_id];
            assert!(block.free_node.is_none());
            block.range.size()
        };
        let free_list_index = Self::free_list_index(size);
        if let Some(next_id) = free_lists[free_list_index] {
            let prev_id = blocks[next_id].free_node.unwrap().prev_id;
            if prev_id == next_id {
                let [other, alloc] = blocks.get_disjoint_mut([prev_id, alloc_id]).unwrap();
                other.free_node = Some(BlockListNode::new(alloc_id));
                alloc.free_node = Some(BlockListNode::new(prev_id));
            } else {
                let [prev, alloc, next] = blocks
                    .get_disjoint_mut([prev_id, alloc_id, next_id])
                    .unwrap();
                prev.free_node.as_mut().unwrap().next_id = alloc_id;
                alloc.free_node = Some(BlockListNode { prev_id, next_id });
                next.free_node.as_mut().unwrap().prev_id = prev_id;
            }
        } else {
            blocks[alloc_id].free_node = Some(BlockListNode::new(alloc_id));
        }
        free_lists[free_list_index] = Some(alloc_id);
    }

    fn unregister_free_block(
        blocks: &mut BlockSlotMap<K, T>,
        free_lists: &mut [Option<K>],
        free_id: K,
    ) {
        let (size, BlockListNode { prev_id, next_id }) = {
            let block = &blocks[free_id];
            (block.range.size(), block.free_node.unwrap())
        };
        let free_list_index = Self::free_list_index(size);
        let head_id = if prev_id == free_id {
            None
        } else if prev_id == next_id {
            blocks[prev_id].free_node = Some(BlockListNode::new(prev_id));
            Some(prev_id)
        } else {
            let [prev, next] = blocks.get_disjoint_mut([prev_id, next_id]).unwrap();
            prev.free_node.as_mut().unwrap().next_id = next_id;
            next.free_node.as_mut().unwrap().prev_id = prev_id;
            Some(next_id)
        };
        free_lists[free_list_index] = head_id;
        blocks[free_id].free_node = None;
    }

    fn truncate_block(blocks: &mut BlockSlotMap<K, T>, orig_id: K, new_size: usize) -> K {
        let (next_id, new_id) = {
            let orig_block = &mut blocks[orig_id];
            let next_id = orig_block.tag_node.next_id;
            let tag = orig_block.tag;
            let range = orig_block.range.truncate(new_size);
            let new_id = blocks.insert_with_key(|key| Block::new(key, tag, range));
            (next_id, new_id)
        };

        if orig_id == next_id {
            let [orig, new] = blocks.get_disjoint_mut([orig_id, new_id]).unwrap();
            orig.tag_node = BlockListNode::new(new_id);
            new.tag_node = BlockListNode::new(orig_id);
        } else {
            let prev_id = orig_id;
            let [prev, new, next] = blocks.get_disjoint_mut([prev_id, new_id, next_id]).unwrap();
            prev.tag_node.next_id = new_id;
            new.tag_node = BlockListNode { prev_id, next_id };
            next.tag_node.prev_id = new_id;
        }

        new_id
    }

    fn append_block(blocks: &mut BlockSlotMap<K, T>, orig_id: K, append_id: K) {
        let [orig_block, append_block] = blocks.get_disjoint_mut([orig_id, append_id]).unwrap();
        orig_block.range.append(append_block.range);

        let next_id = append_block.tag_node.next_id;
        if orig_id == next_id {
            orig_block.tag_node = BlockListNode::new(orig_id);
        } else {
            let [orig_block, next_block] = blocks.get_disjoint_mut([orig_id, next_id]).unwrap();
            orig_block.tag_node.next_id = next_id;
            next_block.tag_node.prev_id = orig_id;
        }

        blocks.remove(append_id).unwrap();
    }

    fn print_state(&self) {
        for (index, first_block_id) in self.free_lists.iter().cloned().enumerate() {
            println!("free list {}:", index);
            if let Some(first_block_id) = first_block_id {
                let mut block_id = first_block_id;
                loop {
                    let block = &self.blocks[block_id];
                    println!("{:?} = {:?}", block_id, block);
                    block_id = block.free_node.unwrap().next_id;
                    if block_id == first_block_id {
                        break;
                    }
                }
            }
        }
        println!("allocated list:");
        for (block_id, block) in self.blocks.iter() {
            if block.free_node.is_none() {
                println!("{:?} = {:?}", block_id, block);
            }
        }
    }

    pub fn alloc(&mut self, size: usize, align: usize) -> Option<K> {
        let blocks = &mut self.blocks;
        let free_lists = self.free_lists.as_mut_slice();

        let align_mask = align - 1;
        let start_free_list_index = Self::free_list_index(size);
        for first_block_id in free_lists
            .get(start_free_list_index..)?
            .iter()
            .cloned()
            .filter_map(|id| id)
        {
            let mut block_id = first_block_id;
            loop {
                let block_range = blocks[block_id].range;
                let aligned_begin = (block_range.begin + align_mask) & !align_mask;
                let aligned_end = aligned_begin + size;
                if aligned_end <= block_range.end {
                    Self::unregister_free_block(blocks, free_lists, block_id);
                    if aligned_begin != block_range.begin {
                        let aligned_id = Self::truncate_block(
                            blocks,
                            block_id,
                            aligned_begin - block_range.begin,
                        );
                        Self::register_free_block(blocks, free_lists, block_id);
                        block_id = aligned_id;
                    }
                    if aligned_end != block_range.end {
                        let unused_id = Self::truncate_block(blocks, block_id, size);
                        Self::register_free_block(blocks, free_lists, unused_id);
                    }
                    return Some(block_id);
                }
                block_id = blocks[block_id].free_node.unwrap().next_id;
                if block_id == first_block_id {
                    break;
                }
            }
        }
        None
    }

    pub fn tag(&self, id: K) -> (T, usize) {
        self.blocks[id].tag()
    }

    pub fn free(&mut self, id: K) {
        let blocks = &mut self.blocks;
        let free_lists = self.free_lists.as_mut_slice();

        let block = &blocks[id];
        assert!(block.free_node.is_none());
        let next_id = block.tag_node.next_id;
        let next = &blocks[next_id];
        if next.free_node.is_some() && block.can_append(next) {
            Self::unregister_free_block(blocks, free_lists, next_id);
            Self::append_block(blocks, id, next_id);
        }

        let block = &blocks[id];
        let prev_id = block.tag_node.prev_id;
        let prev = &blocks[prev_id];
        if prev.free_node.is_some() && prev.can_append(block) {
            Self::unregister_free_block(blocks, free_lists, prev_id);
            Self::append_block(blocks, prev_id, id);
            Self::register_free_block(blocks, free_lists, prev_id);
        } else {
            Self::register_free_block(blocks, free_lists, id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    slotmap::new_key_type! {
        struct Id;
    }
    impl Tag for usize {}

    #[test]
    fn heap_test() {
        let mut heap = Heap::default();
        heap.extend_with(0usize, 1000);

        let ai: Id = heap.alloc(1000, 4).unwrap();
        heap.free(ai);

        let ai = heap.alloc(500, 4).unwrap();
        heap.print_state();
        let bi = heap.alloc(500, 4).unwrap();
        heap.print_state();
        heap.free(ai);
        heap.print_state();
        let ci = heap.alloc(250, 2).unwrap();
        let di = heap.alloc(250, 2).unwrap();
        heap.print_state();
        heap.free(bi);
        heap.print_state();
        heap.free(ci);
        heap.print_state();
        heap.free(di);
        heap.print_state();

        let ei = heap.alloc(1000, 4).unwrap();
        heap.free(ei);
    }
}
