device::Memory* createMemory(size_t size) const override { return nullptr; }
bool createSampler(const Sampler&, device::Sampler**) const override { return false; }
device::Memory* createView( amd::Memory& owner, const device::Memory& parent ) const override { return nullptr; }
device::Signal* createSignal() const override { return nullptr; }
bool bindExternalDevice( uint flags, void* const pDevice[], void* pContext, bool validateOnly ) override { return false; }
bool unbindExternalDevice( uint flags, void* const pDevice[], void* pContext, bool validateOnly ) override { return false; }
bool globalFreeMemory(size_t* freeMemory ) const override { return false; }
bool importExtSemaphore(void** extSemaphore, const amd::Os::FileDesc& handle, amd::ExternalSemaphoreHandleType sem_handle_type) override { return false; }
void DestroyExtSemaphore(void* extSemaphore) override { state_->observed.terminal=true; }
void* svmAlloc(Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags, void* svmPtr) const override { return nullptr; }
void svmFree(void* ptr) const override { state_->observed.terminal=true; }
void* virtualAlloc(void* addr, size_t size, size_t alignment) override { return nullptr; }
bool SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags) override { return false; }
bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) override { return false; }
void virtualFree(void* addr) override { state_->observed.terminal=true; }
#if defined(__clang__)
#if __has_feature(address_sanitizer)
device::UriLocator* createUriLocator() const override { return nullptr; }
#endif
#endif
