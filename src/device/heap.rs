use slotmap::SlotMap;
use std::fmt::Debug;

slotmap::new_key_type! {
    pub struct BlockId;
}

#[derive(Debug, Clone, Copy)]
struct BlockListNode {
    prev_id: BlockId,
    next_id: BlockId,
}

impl BlockListNode {
    fn new(id: BlockId) -> Self {
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

pub trait ArenaId: Debug + Clone + Copy + PartialEq + Eq {}

#[derive(Debug, Clone, Copy)]
struct Block<A: ArenaId> {
    arena: A,
    range: Range,
    arena_node: BlockListNode,
    free_node: Option<BlockListNode>,
}

impl<A: ArenaId> Block<A> {
    fn new(id: BlockId, arena: A, range: Range) -> Self {
        Self {
            arena,
            range,
            arena_node: BlockListNode::new(id),
            free_node: None,
        }
    }

    fn can_append(&self, other: &Block<A>) -> bool {
        self.arena == other.arena && self.range.end == other.range.begin
    }
}

type BlockSlotMap<A> = SlotMap<BlockId, Block<A>>;

#[derive(Debug, Default)]
pub struct Heap<A: ArenaId> {
    blocks: BlockSlotMap<A>,
    free_lists: Vec<Option<BlockId>>,
}

impl<A: ArenaId> Heap<A> {
    fn free_list_index(size: usize) -> usize {
        (0usize.leading_zeros() - size.leading_zeros()) as usize
    }

    pub fn extend_with(&mut self, arena: A, size: usize) {
        let free_list_index = Self::free_list_index(size);

        while free_list_index >= self.free_lists.len() {
            self.free_lists.push(None);
        }

        let id = self
            .blocks
            .insert_with_key(|key| Block::new(key, arena, Range::new(size)));
        Self::register_free_block(&mut self.blocks, self.free_lists.as_mut_slice(), id);
    }

    fn register_free_block(
        blocks: &mut BlockSlotMap<A>,
        free_lists: &mut [Option<BlockId>],
        alloc_id: BlockId,
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
        blocks: &mut BlockSlotMap<A>,
        free_lists: &mut [Option<BlockId>],
        free_id: BlockId,
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

    fn truncate_block(blocks: &mut BlockSlotMap<A>, orig_id: BlockId, new_size: usize) -> BlockId {
        let (next_id, new_id) = {
            let orig_block = &mut blocks[orig_id];
            let next_id = orig_block.arena_node.next_id;
            let arena = orig_block.arena;
            let range = orig_block.range.truncate(new_size);
            let new_id = blocks.insert_with_key(|key| Block::new(key, arena, range));
            (next_id, new_id)
        };

        if orig_id == next_id {
            let [orig, new] = blocks.get_disjoint_mut([orig_id, new_id]).unwrap();
            orig.arena_node = BlockListNode::new(new_id);
            new.arena_node = BlockListNode::new(orig_id);
        } else {
            let prev_id = orig_id;
            let [prev, new, next] = blocks.get_disjoint_mut([prev_id, new_id, next_id]).unwrap();
            prev.arena_node.next_id = new_id;
            new.arena_node = BlockListNode { prev_id, next_id };
            next.arena_node.prev_id = new_id;
        }

        new_id
    }

    fn append_block(blocks: &mut BlockSlotMap<A>, orig_id: BlockId, append_id: BlockId) {
        let [orig_block, append_block] = blocks.get_disjoint_mut([orig_id, append_id]).unwrap();
        orig_block.range.append(append_block.range);

        let next_id = append_block.arena_node.next_id;
        if orig_id == next_id {
            orig_block.arena_node = BlockListNode::new(orig_id);
        } else {
            let [orig_block, next_block] = blocks.get_disjoint_mut([orig_id, next_id]).unwrap();
            orig_block.arena_node.next_id = next_id;
            next_block.arena_node.prev_id = orig_id;
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

    pub fn alloc(&mut self, size: usize, align: usize) -> Option<(BlockId, usize)> {
        let blocks = &mut self.blocks;
        let free_lists = self.free_lists.as_mut_slice();

        let align_mask = align - 1;
        let start_free_list_index = Self::free_list_index(size);
        for first_block_id in free_lists[start_free_list_index..]
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
                    return Some((block_id, aligned_begin));
                }
                block_id = blocks[block_id].free_node.unwrap().next_id;
                if block_id == first_block_id {
                    break;
                }
            }
        }
        None
    }

    pub fn free(&mut self, block_id: BlockId) {
        let blocks = &mut self.blocks;
        let free_lists = self.free_lists.as_mut_slice();

        let block = &blocks[block_id];
        assert!(block.free_node.is_none());
        let next_id = block.arena_node.next_id;
        let next = &blocks[next_id];
        if next.free_node.is_some() && block.can_append(next) {
            Self::unregister_free_block(blocks, free_lists, next_id);
            Self::append_block(blocks, block_id, next_id);
        }

        let block = &blocks[block_id];
        let prev_id = block.arena_node.prev_id;
        let prev = &blocks[prev_id];
        if prev.free_node.is_some() && prev.can_append(block) {
            Self::unregister_free_block(blocks, free_lists, prev_id);
            Self::append_block(blocks, prev_id, block_id);
            Self::register_free_block(blocks, free_lists, prev_id);
        } else {
            Self::register_free_block(blocks, free_lists, block_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl ArenaId for usize {}

    #[test]
    fn heap_test() {
        let mut heap = Heap::default();
        heap.extend_with(0usize, 1000);

        let (ai, _) = heap.alloc(1000, 4).unwrap();
        heap.free(ai);

        let (ai, _) = heap.alloc(500, 4).unwrap();
        heap.print_state();
        let (bi, _) = heap.alloc(500, 4).unwrap();
        heap.print_state();
        heap.free(ai);
        heap.print_state();
        let (ci, _) = heap.alloc(250, 2).unwrap();
        let (di, _) = heap.alloc(250, 2).unwrap();
        heap.print_state();
        heap.free(bi);
        heap.print_state();
        heap.free(ci);
        heap.print_state();
        heap.free(di);
        heap.print_state();

        let (ei, _) = heap.alloc(1000, 4).unwrap();
        heap.free(ei);
    }
}
