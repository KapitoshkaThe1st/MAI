#include "Allocator.h"

Allocator::Allocator(size_t size, size_t block_size) : m_size(size) {
	m_memory = static_cast<char*>(malloc(sizeof(char) * size));
	size_t blocks_count = size / block_size;
	m_blocks = std::move(Array<char*>(blocks_count));
	for (size_t i = 0; i < blocks_count; ++i)
		m_blocks[i] = m_memory + i * block_size;
	m_free_blocks_count = blocks_count;
}
Allocator::~Allocator() {
	free(m_memory);
}
void* Allocator::allocate() {
	if (m_free_blocks_count > 0) {
		void *result = m_blocks[m_free_blocks_count - 1];
		--m_free_blocks_count;
		return static_cast<void*>(result);
	}
	return nullptr;
}
void Allocator::deallocate(void *to_deallocate) {
	++m_free_blocks_count;
	m_blocks[m_free_blocks_count - 1] = static_cast<char*>(to_deallocate);
}
bool Allocator::has_free_blocks() {
	return m_free_blocks_count > 0;
}
size_t Allocator::blocks_available() {
	return m_free_blocks_count;
}