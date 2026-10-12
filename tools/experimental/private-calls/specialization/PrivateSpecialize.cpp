// Isolated research pass. No compiler/runtime installation or name-based fixture handling.
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Operator.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <map>
using namespace llvm;
namespace {
bool allowedPointerAttrs(AttributeSet AS) {
  for (Attribute A:AS) {
    if (A.isStringAttribute()) return false;
    switch (A.getKindAsEnum()) {
    case Attribute::NoCapture:
    case Attribute::NoUndef:
    case Attribute::ReadOnly:
    case Attribute::WriteOnly: break;
    default: return false; // ABI conventions and unreviewed pointer facts stay generic.
    }
  }
  return true;
}
bool eligible(Function &F, unsigned Index) {
  if (!F.hasLocalLinkage() || F.isDeclaration() || F.isVarArg() || F.hasAddressTaken() ||
      !F.doesNotRecurse() || !F.getReturnType()->isIntegerTy() ||
      F.hasFnAttribute(Attribute::Convergent) || F.hasFnAttribute(Attribute::OptimizeNone) ||
      !F.getArg(Index)->hasNoCaptureAttr() || !allowedPointerAttrs(F.getAttributes().getParamAttrs(Index))) return false;
  SmallVector<Value *,32> Work{F.getArg(Index)}; SmallPtrSet<Value *,32> Seen;
  while (!Work.empty()) {
    Value *V=Work.pop_back_val(); if (!Seen.insert(V).second) continue;
    for (User *U:V->users()) {
      if (auto *G=dyn_cast<GetElementPtrInst>(U)) { if(G->getPointerOperand()!=V) return false; Work.push_back(G); }
      else if (auto *B=dyn_cast<BitCastInst>(U)) Work.push_back(B);
      else if (auto *L=dyn_cast<LoadInst>(U)) { if(L->getPointerOperand()!=V || !L->getType()->isIntegerTy(32) || L->isAtomic()) return false; }
      else if (auto *S=dyn_cast<StoreInst>(U)) { if(S->getPointerOperand()!=V || !S->getValueOperand()->getType()->isIntegerTy(32) || S->isAtomic()) return false; }
      else return false; // No escapes, comparisons, casts to integers, phi/select or forwarded argument yet.
    }
  }
  return true;
}
struct PrivateSpecialize:PassInfoMixin<PrivateSpecialize> {
 PreservedAnalyses run(Module &M,ModuleAnalysisManager &) {
  if (M.getTargetTriple()!="amdgcn-amd-amdhsa") return PreservedAnalyses::all();
  std::map<std::pair<Function *,unsigned>,Function *> Clones; bool Changed=false;
  std::map<Function *,Function *> Origins;
  std::map<Function *,unsigned> Descendants;
  unsigned ModuleClones=0;
  constexpr unsigned MaxModuleClones=64, MaxOriginClones=8;
  for(unsigned Round=0;Round<8;++Round) {
   SmallVector<CallInst *,32> Calls;
   for(Function &F:M) for(Instruction &I:instructions(F)) if(auto *C=dyn_cast<CallInst>(&I)) Calls.push_back(C);
   bool Again=false;
   for(CallInst *C:Calls) {
    Function *F=C->getCalledFunction();
    if(!F || F->isVarArg() || C->isMustTailCall() || C->hasOperandBundles() || F->isIntrinsic()) continue;
    for(unsigned A=0;A<C->arg_size();++A) {
     auto *PT=dyn_cast<PointerType>(F->getArg(A)->getType()); if(!PT || PT->getAddressSpace()!=0) continue;
     auto *Cast=dyn_cast<AddrSpaceCastOperator>(C->getArgOperand(A));
     if(!Cast || Cast->getSrcAddressSpace()!=5 || Cast->getDestAddressSpace()!=0 || !allowedPointerAttrs(C->getAttributes().getParamAttrs(A)) || !eligible(*F,A)) continue;
     Function *&N=Clones[{F,A}];
     if(!N) {
      Function *Origin=Origins.count(F) ? Origins.at(F) : F;
      // Experimental per-invocation resource bounds; leave this call unchanged when full.
      if(ModuleClones>=MaxModuleClones || Descendants[Origin]>=MaxOriginClones) continue;
      SmallVector<Type *,8> Types(F->getFunctionType()->params()); Types[A]=PointerType::get(M.getContext(),5);
      N=Function::Create(FunctionType::get(F->getReturnType(),Types,false),GlobalValue::InternalLinkage,F->getName()+".private5",M);
      Origins[N]=Origin; ++Descendants[Origin]; ++ModuleClones;
      N->copyAttributesFrom(F); N->setCallingConv(F->getCallingConv());
      auto *Entry=BasicBlock::Create(M.getContext(),"private.entry",N); IRBuilder<> B(Entry);
      ValueToValueMapTy VM;
      for(unsigned I=0;I<F->arg_size();++I) VM[F->getArg(I)]=I==A ? B.CreateAddrSpaceCast(N->getArg(I),F->getArg(I)->getType()) : N->getArg(I);
      SmallVector<ReturnInst *,8> Returns;
      CloneFunctionInto(N,F,VM,CloneFunctionChangeType::LocalChangesOnly,Returns);
      B.CreateBr(cast<BasicBlock>(VM[&F->getEntryBlock()]));
     }
     SmallVector<Value *,8> Args(C->args()); Args[A]=Cast->getOperand(0);
     auto *New=CallInst::Create(N,Args,"",C); New->setCallingConv(C->getCallingConv()); New->setTailCallKind(C->getTailCallKind()); New->setAttributes(C->getAttributes()); New->setDebugLoc(C->getDebugLoc());
     C->replaceAllUsesWith(New); C->eraseFromParent(); Changed=Again=true; break;
    }
   }
   if(!Again) break;
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
 }
};
}
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
 return {LLVM_PLUGIN_API_VERSION,"research-private-specialize","0.1",[](PassBuilder &PB){
  PB.registerPipelineParsingCallback([](StringRef Name,ModulePassManager &MPM,ArrayRef<PassBuilder::PipelineElement>){
   if(Name!="research-private-specialize") return false; MPM.addPass(PrivateSpecialize()); return true;
  });
 }};
}



