"""Bounded transfers; caller retains existing idle/locks/promotion policy."""
import hashlib
import time
from pathlib import Path
CHUNK = 65536

def require(ok, why):
    if not ok: raise ValueError(why)

def local_identity(path, size, digest):
    h=hashlib.sha256(); count=0
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(CHUNK), b''):
            count+=len(block); require(count<=size, 'local oversize'); h.update(block)
    require(count==size and h.hexdigest()==digest, 'local identity mismatch')

class BoundedReader:
    def __init__(self, stream, size, deadline, guard=lambda:None):
        self.guard=guard; self.stream=stream; self.size=size; self.deadline=deadline; self.count=0; self.hash=hashlib.sha256()
    def read(self, n=CHUNK):
        self.guard()
        require(0<n<=CHUNK, 'unbounded upload read')
        require(time.monotonic()<self.deadline, 'upload deadline')
        block=self.stream.read(n); self.count+=len(block)
        require(self.count<=self.size, 'upload oversize'); self.hash.update(block)
        return block

def readback(ftp, remote, size, digest, records, seconds, output=None, guard=lambda:None):
    require(type(size) is int and size>0, 'invalid exact size')
    require(0<seconds<=1800, 'invalid explicit transfer deadline')
    record={'path':remote,'expected_bytes':size,'expected_sha256':digest,'complete':False,'received_bytes':0}
    h=hashlib.sha256(); started=time.monotonic(); sink=None
    try:
        if output is not None: sink=Path(output).open('xb')
        def consume(block):
            guard()
            require(len(block)<=CHUNK, 'oversize transport chunk')
            record['received_bytes']+=len(block)
            require(record['received_bytes']<=size, 'remote oversize')
            require(time.monotonic()-started<seconds, 'readback deadline')
            h.update(block)
            if sink is not None: sink.write(block)
        ftp.retrbinary('RETR '+remote,consume,blocksize=CHUNK)
        require(time.monotonic()-started<seconds, 'readback deadline')
        require(record['received_bytes']==size and h.hexdigest()==digest, 'remote identity mismatch')
        record['complete']=True
    except Exception as error:
        record['error']=repr(error); raise
    finally:
        if sink is not None: sink.close()
        record['received_sha256']=h.hexdigest();record['elapsed_seconds']=time.monotonic()-started;records.append(record)
    return record

def stage_upload(ftp, local, temporary, size, digest, records, seconds, guard=lambda:None):
    """Never rename; existing controller promotes only after all lifecycle checks."""
    require(0<seconds<=1800, 'invalid explicit transfer deadline')
    local_identity(local,size,digest)
    started=time.monotonic(); record={'path':temporary,'complete':False,'expected_bytes':size,'expected_sha256':digest}
    reader=None
    try:
        with Path(local).open('rb') as stream:
            reader=BoundedReader(stream,size,started+seconds,guard)
            ftp.storbinary('STOR '+temporary,reader,blocksize=CHUNK)
        require(time.monotonic()-started<seconds, 'upload deadline')
        require(reader.count==size and reader.hash.hexdigest()==digest, 'upload identity mismatch')
        local_identity(local,size,digest)
        record['complete']=True
    except Exception as error:
        record['error']=repr(error);raise
    finally:
        record['sent_bytes']=reader.count if reader else 0
        record['sent_sha256']=reader.hash.hexdigest() if reader else None
        record['elapsed_seconds']=time.monotonic()-started;records.append(record)
    guard()
    remaining=seconds-(time.monotonic()-started)
    require(remaining>0, 'upload/readback deadline')
    return readback(ftp,temporary,size,digest,records,remaining,guard=guard)
