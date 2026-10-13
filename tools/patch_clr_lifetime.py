"""Apply checked POSIX lifetime fixes to an isolated ROCclr source copy."""
import re
def patch(source):
 changed=[]
 def edit(name,old,new):
  p=source/'rocclr'/name;t=p.read_text();assert t.count(old)==1,(name,old);p.write_text(t.replace(old,new));changed.append(p)
 edit('thread/thread.hpp','volatile ThreadState state_;','std::atomic<ThreadState> state_;')
 edit('thread/thread.hpp','void setState(ThreadState state) { state_ = state; }','void setState(ThreadState state) { state_.store(state, std::memory_order_release); }')
 edit('thread/thread.hpp','ThreadState state() const { return state_; }','ThreadState state() const { return state_.load(std::memory_order_acquire); }')
 edit('thread/thread.hpp','  bool start(void* data = NULL);','  bool start(void* data = NULL);\n#if defined(PS5HIP_CLR_PLATFORM)\n  bool join(); // Single external owner; failed join must retain the enclosing queue.\n#endif')
 edit('os/os_posix.cpp','  // We never plan the use join, so free the resources now.\n  ::pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_DETACHED);','''#if !defined(PS5HIP_CLR_PLATFORM)
  ::pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_DETACHED);
#endif''')
 edit('os/os_posix.cpp','bool Os::isThreadAlive(const Thread& thread) {','''#if defined(PS5HIP_CLR_PLATFORM)
bool Thread::join() {
  if (handle_ == nullptr) return true;
  const pthread_t target = (pthread_t)handle_;
  if (::pthread_equal(target, ::pthread_self())) return false;
  if (::pthread_join(target, nullptr) != 0) return false;
  handle_ = nullptr; // Never reuse a consumed pthread identity.
  return true;
}
#endif

bool Os::isThreadAlive(const Thread& thread) {
#if defined(PS5HIP_CLR_PLATFORM)
  return thread.handle() != nullptr && thread.state() < Thread::FINISHED;
#else''')
 edit('os/os_posix.cpp','  return ::pthread_kill((pthread_t)thread.handle(), 0) == 0;','  return ::pthread_kill((pthread_t)thread.handle(), 0) == 0;\n#endif')
 edit('platform/commandqueue.cpp','  if (Agent::shouldPostCommandQueueEvents()) {\n    Agent::postCommandQueueFree','''#if defined(PS5HIP_CLR_PLATFORM)
  if (!AMD_DIRECT_DISPATCH && !thread_.join()) return false;
#endif
  if (Agent::shouldPostCommandQueueEvents()) {
    Agent::postCommandQueueFree''')
 edit('platform/commandqueue.cpp','if (thread_.state() >= Thread::INITIALIZED)', 'if (thread_.state() == Thread::INITIALIZED)')
 edit('platform/commandqueue.cpp','bool HostQueue::terminate() {','''bool HostQueue::terminate() {
#if defined(PS5HIP_CLR_PLATFORM)
  // A worker cannot wait for its own marker or completion. Retain the queue.
  if (!AMD_DIRECT_DISPATCH && amd::Thread::current() == &thread_) return false;
#endif''')
 edit('platform/runtime.cpp','referenceCount_.fetch_sub(1, std::memory_order_relaxed)','referenceCount_.fetch_sub(1, std::memory_order_acq_rel)')
 edit('thread/semaphore.hpp','  Semaphore();','''  // HeapObject's malloc overload does not honor this class's 64-byte alignment.
  void* operator new(size_t n) { void* p = AlignedMemory::allocate(n, 64); if (!p) throw std::bad_alloc(); return p; }
  void operator delete(void* p) { AlignedMemory::deallocate(p); }
  Semaphore();''')
 edit('thread/semaphore.hpp','#include "utils/util.hpp"','#include "utils/util.hpp"\n#include "os/alloc.hpp"\n#include <new>')
 edit('os/alloc.cpp','void* HeapObject::operator new(size_t size) { return malloc(size); }','void* HeapObject::operator new(size_t size) { void* p = malloc(size); if (!p) throw std::bad_alloc(); return p; }')
 edit('os/alloc.cpp','#include "os/alloc.hpp"','#include "os/alloc.hpp"\n#include <new>')
 edit('thread/thread.cpp','''  created_ = new Semaphore();
  lock_ = new Semaphore();
  suspend_ = new Semaphore();

  selfSuspendLock_ = new Monitor();''','''  created_ = lock_ = suspend_ = nullptr;
  selfSuspendLock_ = nullptr;
  try {
    created_ = new Semaphore();
    lock_ = new Semaphore();
    suspend_ = new Semaphore();
    selfSuspendLock_ = new Monitor();
  } catch (...) {
    delete selfSuspendLock_;
    delete suspend_;
    delete lock_;
    delete created_;
    throw; // No OS thread exists until create() returns successfully.
  }''')
 # Tagged values are not objects, and may be null. Extract through static
 # functions instead of invoking a method on a fictitious object address.
 p=source/'rocclr/utils/concurrent.hpp';t=p.read_text()
 t=t.replace('T* ptr() { return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(this) & ~TagMask); }','static T* ptr(const TaggedPointerHelper* value) { return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(value) & ~TagMask); }')
 t=t.replace('size_t tag() const { return reinterpret_cast<uintptr_t>(this) & TagMask; }','static size_t tag(const TaggedPointerHelper* value) { return reinterpret_cast<uintptr_t>(value) & TagMask; }')
 t=re.sub(r'\b(head|tail|next)->(ptr|tag)\(\)',r'Node::TaggedPointerHelper::\2(\1)',t)
 p.write_text(t);changed.append(p)
 return list(dict.fromkeys(changed))
