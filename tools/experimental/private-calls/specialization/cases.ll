target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define internal fastcc zeroext i32 @pair(ptr nocapture readonly %p, ptr nocapture %q, i32 %index) noinline norecurse nounwind {
  %at = getelementptr inbounds i32, ptr %p, i32 %index
  %x = load i32, ptr %at, align 4
  %y = load volatile i32, ptr %q, align 4
  %sum = add i32 %x, %y
  store volatile i32 %sum, ptr %q, align 4
  ret i32 %sum
}

define i32 @many_callers(ptr addrspace(5) %p, ptr addrspace(5) %q, ptr addrspace(1) %g, i32 %i) {
  %a = addrspacecast ptr addrspace(5) %p to ptr
  %b = addrspacecast ptr addrspace(5) %q to ptr
  %c = addrspacecast ptr addrspace(1) %g to ptr
  %one = call fastcc zeroext i32 @pair(ptr %a, ptr %c, i32 %i)
  %two = call fastcc zeroext i32 @pair(ptr %c, ptr %b, i32 %i)
  %both = call fastcc zeroext i32 @pair(ptr %a, ptr %b, i32 %i)
  %repeat = call fastcc zeroext i32 @pair(ptr %a, ptr %b, i32 %i)
  %global = call fastcc zeroext i32 @pair(ptr %c, ptr %c, i32 %i)
  %r = add i32 %both, %global
  ret i32 %r
}

define internal i32 @unused(ptr nocapture %p) noinline norecurse {
  ret i32 7
}
define i32 @poison_null() {
  %p = addrspacecast ptr addrspace(5) poison to ptr
  %n = addrspacecast ptr addrspace(5) null to ptr
  %u = addrspacecast ptr addrspace(5) undef to ptr
  %a = call i32 @unused(ptr %p)
  %b = call i32 @unused(ptr %n)
  %c = call i32 @unused(ptr %u)
  %ab = add i32 %a, %b
  %abc = add i32 %ab, %c
  ret i32 %abc
}

declare i32 @unknown(ptr nocapture)
define internal i32 @forward(ptr nocapture %p) noinline norecurse {
  %x = call i32 @unknown(ptr %p)
  ret i32 %x
}
define internal i32 @escape(ptr %p, ptr %dst) noinline norecurse {
  store ptr %p, ptr %dst
  ret i32 9
}
define internal i32 @ptrcompare(ptr nocapture %p) noinline norecurse {
  %x = icmp eq ptr %p, null
  %r = zext i1 %x to i32
  ret i32 %r
}
define internal i32 @selectptr(ptr nocapture %p, ptr %g, i1 %which) noinline norecurse {
  %x = select i1 %which, ptr %p, ptr %g
  %r = load i32, ptr %x
  ret i32 %r
}
define i32 @guards(ptr addrspace(5) %p, ptr %g, i1 %which) {
  %a = addrspacecast ptr addrspace(5) %p to ptr
  %f = call i32 @forward(ptr %a)
  %e = call i32 @escape(ptr %a, ptr %g)
  %c = call i32 @ptrcompare(ptr %a)
  %s = call i32 @selectptr(ptr %a, ptr %g, i1 %which)
  ret i32 %f
}

define internal i32 @offset(ptr nocapture readonly %p, i64 %n) noinline norecurse {
  %p1 = getelementptr i8, ptr %p, i64 %n
  %p2 = getelementptr inbounds i32, ptr %p1, i32 1
  %r = load i32, ptr %p2, align 4
  ret i32 %r
}
define i32 @offset_caller(ptr addrspace(5) %p, i64 %n) {
  %a = addrspacecast ptr addrspace(5) %p to ptr
  %r = call i32 @offset(ptr %a, i64 %n)
  ret i32 %r
}
