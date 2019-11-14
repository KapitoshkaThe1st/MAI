#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <cstdlib>
#include <algorithm>

#include "Array.h"

class Allocator {
public:
	Allocator(size_t size, size_t block_size);
	~Allocator();
	void* allocate();
	void deallocate(void *to_deallocate);
	bool has_free_blocks();
	size_t blocks_available();
private:
	Array<char*> m_blocks;
	size_t m_free_blocks_count;
	size_t m_size;
	char *m_memory;
};
#endif // !ALLOCATOR_H