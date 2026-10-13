"""Real event publication and final queue batch retirement, preserving v6."""
from patch_clr_lifetime_v6 import patch as previous
def patch(source):
 changed=previous(source)
 p=source/'rocclr/platform/command.hpp';t=p.read_text()
 old='int32_t status() const { return status_.load(std::memory_order_relaxed); }'
 assert t.count(old)==1;t=t.replace(old,old.replace('relaxed','acquire'));p.write_text(t);changed.append(p)
 p=source/'rocclr/platform/command.cpp';t=p.read_text()
 old='status_.compare_exchange_strong(currentStatus, status, std::memory_order_relaxed)'
 assert t.count(old)==3;t=t.replace(old,old.replace('relaxed','acq_rel'));p.write_text(t);changed.append(p)
 p=source/'rocclr/platform/commandqueue.cpp';t=p.read_text()
 old='''        if (!thread_.acceptingCommands_) {
          return;
        }'''
 assert t.count(old)==1;t=t.replace(old,old.replace('return;','break;'))
 old='''    command->retain();

    // Process the command's event wait list.'''
 new='''    if (command == nullptr) {
      // Even already-completed synchronous commands retain a batch reference.
      // Retire it before destroying the virtual device, outside queueLock_.
      virtualDevice->flush(head, true);
      return;
    }
    command->retain();

    // Process the command's event wait list.'''
 assert t.count(old)==1;t=t.replace(old,new);p.write_text(t)
 return list(dict.fromkeys(changed))
