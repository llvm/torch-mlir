//===- CPASupport.h - Support types and utilities for CPA -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support types and utilities for the Cartesian Product Algorithm for
// Type Inference.
//
// See:
//   http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.30.8177
//   http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.129.2756
//===----------------------------------------------------------------------===//

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#ifndef NPCOMP_TYPING_ANALYSIS_CPA_SUPPORT_H
#define NPCOMP_TYPING_ANALYSIS_CPA_SUPPORT_H

namespace mlir {
namespace NPCOMP {
namespace Typing {
namespace CPA {

class Context;
class TypeVarSet;
class TypeVarMap;

/// A uniqued string identifier.
class Identifier {
public:
  StringRef getValue() const { return value; }

  void print(raw_ostream &os, bool brief = false);

  friend raw_ostream &operator<<(raw_ostream &os, Identifier ident) {
    ident.print(os);
    return os;
  }

private:
  Identifier(StringRef value) : value(value) {}
  StringRef value;
  friend class Context;
};

/// Base class for the CPA type hierarchy.
class ObjectBase {
public:
  enum class Kind {
    // Type
    FIRST_TYPE,
    TypeNode = FIRST_TYPE,
    TypeVar,
    CastType,
    ReadType,
    WriteType,

    // ValueType
    FIRST_VALUE_TYPE,
    ValueType = FIRST_VALUE_TYPE,
    IRValueType,
    ObjectValueType,
    LAST_VALUE_TYPE = ObjectValueType,

    LAST_TYPE = TypeVar,

    // Constraint
    Constraint,
  };
  ObjectBase(Kind kind) : kind(kind) {}
  virtual ~ObjectBase();

  Kind getKind() const { return kind; }

  virtual void print(Context &context, raw_ostream &os, bool brief = false) = 0;

private:
  const Kind kind;
};

/// Base class for types.
/// This type hierarchy is adapted from section 2.1 of:
///   Precise Constraint-Based Type Inference for Java
///
/// Referred to as: 'τ' (tau)
class TypeNode : public ObjectBase {
public:
  TypeNode(Kind kind, unsigned hashValue)
      : ObjectBase(kind),
        hashValue(llvm::hash_combine(static_cast<int>(kind), hashValue)) {}
  static bool classof(const ObjectBase *tb) {
    return tb->getKind() >= Kind::FIRST_TYPE &&
           tb->getKind() <= Kind::LAST_TYPE;
  }

  /// Collects all type variables that are dependencies of this TypeNode.
  virtual void collectDependentTypeVars(TypeVarSet &typeVars);

  /// Constructs a corresponding IR type for this TypeNode.
  /// Returns a null Type on error, optionally emitting an error if a Location
  /// is provided.
  /// Not all TypeNodes in all states can be converted back to an IR type.
  virtual mlir::Type constructIrType(const TypeVarMap &mapping,
                                     MLIRContext *mlirContext,
                                     llvm::Optional<Location> loc = llvm::None);

  bool operator==(const TypeNode &that) const;
  void print(Context &context, raw_ostream &os, bool brief = false) override;

  struct PtrInfo : llvm::DenseMapInfo<TypeNode *> {
    static TypeNode *getEmptyKey() {
      static TypeNode empty(Kind::TypeNode, 0);
      return &empty;
    }
    static TypeNode *getTombstoneKey() {
      static TypeNode tombstone(Kind::TypeNode, 1);
      return &tombstone;
    }
    static unsigned getHashValue(TypeNode *key) { return key->hashValue; }
    static bool isEqual(TypeNode *lhs, TypeNode *rhs) {
      if (lhs->getKind() == Kind::TypeNode ||
          rhs->getKind() == Kind::TypeNode) {
        // Base class is only created for special static values.
        return lhs == rhs;
      }
      if (lhs == rhs)
        return true;
      return *lhs == *rhs;
    }
  };

private:
  unsigned hashValue;

  friend struct PtrInfo;
};

/// A unique type variable.
/// Both the pointer and the ordinal will be unique within a context.
/// Referred to as 't'
class TypeVar : public TypeNode {
public:
  static bool classof(const ObjectBase *tb) {
    return tb->getKind() == Kind::TypeVar;
  }

  int getOrdinal() const { return ordinal; }

  void print(Context &context, raw_ostream &os, bool brief = false) override;

  /// Constructs a corresponding IR type for this TypeNode.
  /// Returns a null Type on error, optionally emitting an error if a Location
  /// is provided.
  /// Not all TypeNodes in all states can be converted back to an IR type.
  /// Note that this facility is insufficient for the construction of
  /// recursive types (which are presently excluded from being represented
  /// at all).
  mlir::Type
  constructIrType(const TypeVarMap &mapping, MLIRContext *mlirContext,
                  llvm::Optional<Location> loc = llvm::None) override;

private:
  TypeVar(int ordinal)
      : TypeNode(Kind::TypeVar, llvm::hash_code(ordinal)), ordinal(ordinal) {}
  int ordinal;
  friend class Context;
};

/// A type-cast type.
/// Referred to as: 'cast(δ, t)'
class CastType : public TypeNode {
public:
  static bool classof(const ObjectBase *tb) {
    return tb->getKind() == Kind::CastType;
  }

  Identifier *getTypeIdentifier() const { return typeIdentifier; }
  TypeVar *getTypeVar() const { return typeVar; }

  void print(Context &context, raw_ostream &os, bool brief = false) override;

private:
  CastType(Identifier *typeIdentifier, TypeVar *typeVar)
      : TypeNode(Kind::CastType, llvm::hash_combine(typeIdentifier, typeVar)),
        typeIdentifier(typeIdentifier), typeVar(typeVar) {}
  Identifier *typeIdentifier;
  TypeVar *typeVar;
  friend class Context;
};

/// Type representing a read-field operation.
/// Referred to as: 'read τ'
class ReadType : public TypeNode {
public:
  static bool classof(const ObjectBase *tb) {
    return tb->getKind() == Kind::ReadType;
  }

  TypeNode *getType() const { return type; }

  void print(Context &context, raw_ostream &os, bool brief = false) override;

private:
  ReadType(TypeNode *type)
      : TypeNode(Kind::ReadType, llvm::hash_combine(type)), type(type) {}
  TypeNode *type;
  friend class Context;
};

/// Type representing a read-field operation.
/// Referred to as: 'read τ'
class WriteType : public TypeNode {
public:
  static bool classof(const ObjectBase *tb) {
    return tb->getKind() == Kind::WriteType;
  }

  TypeNode *getType() const { return type; }

  void print(Context &context, raw_ostream &os, bool brief = false) override;

private:
  WriteType(TypeNode *type)
      : TypeNode(Kind::WriteType, llvm::hash_combine(type)), type(type) {}
  TypeNode *type;
  friend class Context;
};

/// A legal value type in the language. We represent this as one of:
///   IRValueType: Wraps a primitive MLIR type
///   ObjectValueType: Defines an object.
/// Referred to as 'τv' (tau-v)
class ValueType : public TypeNode {
public:
  using TypeNode::TypeNode;
  static bool classof(const ObjectBase *ob) {
    return ob->getKind() >= Kind::FIRST_VALUE_TYPE &&
           ob->getKind() <= Kind::LAST_VALUE_TYPE;
  }
};

/// Concrete ValueType that wraps an MLIR Type.
class IRValueType : public ValueType {
public:
  IRValueType(mlir::Type irType)
      : ValueType(Kind::IRValueType, llvm::hash_combine(irType)),
        irType(irType) {}
  static bool classof(const ObjectBase *ob) {
    return ob->getKind() == Kind::IRValueType;
  }

  mlir::Type getIrType() const { return irType; }

  void print(Context &context, raw_ostream &os, bool brief = false) override;
  mlir::Type
  constructIrType(const TypeVarMap &mapping, MLIRContext *mlirContext,
                  llvm::Optional<Location> loc = llvm::None) override;

private:
  const mlir::Type irType;
};

/// ValueType for an object.
/// Referred to as 'obj(δ, [ li : τi ])'
class ObjectValueType : public ValueType {
public:
  /// Constructs a corresponding IR type given a list of resolved field types.
  using IrTypeConstructor =
      std::function<mlir::Type(ObjectValueType *ovt, llvm::ArrayRef<mlir::Type>,
                               MLIRContext *, llvm::Optional<Location>)>;

  static bool classof(const ObjectBase *ob) {
    return ob->getKind() == Kind::ObjectValueType;
  }

  Identifier *getTypeIdentifier() { return typeIdentifier; }
  size_t getFieldCount() { return fieldCount; }
  llvm::ArrayRef<Identifier *> getFieldIdentifiers() {
    return llvm::ArrayRef<Identifier *>(fieldIdentifiers, fieldCount);
  }
  llvm::ArrayRef<TypeNode *> getFieldTypes() {
    return llvm::ArrayRef<TypeNode *>(fieldTypes, fieldCount);
  }

  void print(Context &context, raw_ostream &os, bool brief = false) override;
  void collectDependentTypeVars(TypeVarSet &typeVars) override;
  mlir::Type
  constructIrType(const TypeVarMap &mapping, MLIRContext *mlirContext,
                  llvm::Optional<Location> loc = llvm::None) override;

private:
  ObjectValueType(IrTypeConstructor irCtor, Identifier *typeIdentifier,
                  size_t fieldCount, Identifier *const *fieldIdentifiers,
                  TypeNode *const *fieldTypes)
      // TODO: Real hashcode.
      : ValueType(Kind::ObjectValueType, 0), irCtor(std::move(irCtor)),
        typeIdentifier(typeIdentifier), fieldCount(fieldCount),
        fieldIdentifiers(fieldIdentifiers), fieldTypes(fieldTypes) {}
  IrTypeConstructor irCtor;
  Identifier *typeIdentifier;
  size_t fieldCount;
  Identifier *const *fieldIdentifiers;
  TypeNode *const *fieldTypes;
  friend class Context;
};

/// A Constraint between two types.
/// Referred to as: 'τ1 <: τ2'
class Constraint : public ObjectBase {
public:
  static bool classof(ObjectBase *ob) {
    return ob->getKind() == Kind::Constraint;
  }

  TypeNode *getFrom() { return from; }
  TypeNode *getTo() { return to; }

  void print(Context &context, raw_ostream &os, bool brief = false) override;

  bool operator==(const Constraint &that) const {
    return from == that.from && to == that.to;
  }

  struct PtrInfo : llvm::DenseMapInfo<TypeNode *> {
    static Constraint *getEmptyKey() {
      auto emptyType = TypeNode::PtrInfo::getEmptyKey();
      static Constraint empty(emptyType, emptyType);
      return &empty;
    }
    static Constraint *getTombstoneKey() {
      auto tombstoneType = TypeNode::PtrInfo::getTombstoneKey();
      static Constraint tombstone(tombstoneType, tombstoneType);
      return &tombstone;
    }
    static unsigned getHashValue(Constraint *key) {
      return llvm::hash_combine(key->from, key->to);
    }
    static bool isEqual(Constraint *lhs, Constraint *rhs) {
      return *lhs == *rhs;
    }
  };

private:
  Constraint(TypeNode *from, TypeNode *to)
      : ObjectBase(Kind::Constraint), from(from), to(to) {}
  TypeNode *from;
  TypeNode *to;
  friend class Context;
};

/// A set of constraints.
/// Referred to as: 'C'
class ConstraintSet : public llvm::SmallPtrSet<Constraint *, 4> {
public:
  static const ConstraintSet &getEmptySet();
  using SmallPtrSet::SmallPtrSet;
  void print(Context &context, raw_ostream &os, bool brief = false);
};

/// A set of TypeVar.
/// Referred to as 't_bar'
class TypeVarSet : public llvm::SmallPtrSet<TypeVar *, 4> {
public:
  static const TypeVarSet &getEmptySet();
  using SmallPtrSet::SmallPtrSet;
  void print(Context &context, raw_ostream &os, bool brief = false);
};

/// A small mapping of TypeVar -> TypeNode.
class TypeVarMap : public llvm::SmallMapVector<TypeVar *, TypeNode *, 4> {};

/// Set for managing TypeNodes.
class TypeNodeSet : public llvm::SmallPtrSet<TypeNode *, 4> {
public:
  static const TypeNodeSet &getEmptySet();
  using SmallPtrSet::SmallPtrSet;
};

/// Set for managing ValueTypes associated with a TypeVar.
class ValueTypeSet : public llvm::SmallPtrSet<ValueType *, 4> {
public:
  static const ValueTypeSet &getEmptySet();
  using SmallPtrSet::SmallPtrSet;
};

/// Represents an evaluation scope (i.e. a "countour" in the literature) that
/// tracks type variables, IR associations and constraints.
class Environment {
public:
  Environment(Context &context);

  Context &getContext() { return context; }
  ConstraintSet &getConstraints() { return constraints; }
  TypeVarSet &getTypeVars() { return typeVars; }

  /// Maps an IR value to a CPA type by applying an IR Type -> CPA Type
  /// transfer function if not already mapped.
  TypeNode *mapValueToType(Value value);

private:
  Context &context;
  ConstraintSet constraints;
  TypeVarSet typeVars;
  llvm::DenseMap<Value, TypeNode *> valueTypeMap;
};

/// Manages instances and containers needed for the lifetime of a CPA
/// analysis.
class Context {
public:
  /// Hook for customizing default IR Type -> TypeNode conversion.
  /// This is run if more specific conversions fail.
  using IrTypeMapHook =
      std::function<TypeNode *(Context &context, ::mlir::Type irType)>;

  Context(IrTypeMapHook irTypeMapHook = nullptr);

  /// Gets the current environment (roughly call scope).
  Environment &getCurrentEnvironment() { return *currentEnvironment; }

  /// Maps an IR Type to a CPA TypeNode.
  /// This is currently not overridable but a hook may need to be provided
  /// eventually.
  TypeNode *mapIrType(::mlir::Type irType);

  // Create a new (non-uniqued) type var. These are not uniqued because by
  // construction, we only ever ask for new type variables when needed.
  TypeVar *newTypeVar() {
    TypeVar *tv = allocator.Allocate<TypeVar>(1);
    new (tv) TypeVar(++typeVarCounter);
    currentEnvironment->getTypeVars().insert(tv);
    return tv;
  }

  /// Gets a uniqued Identifier for the given value.
  Identifier *getIdentifier(StringRef value) {
    auto it = identifierMap.find(value);
    if (it != identifierMap.end())
      return it->second;
    auto *chars = allocator.Allocate<char>(value.size());
    std::memcpy(chars, value.data(), value.size());
    StringRef uniquedValue(chars, value.size());
    Identifier *id = allocator.Allocate<Identifier>(1);
    new (id) Identifier(uniquedValue);
    identifierMap[uniquedValue] = id;
    return id;
  }

  /// Gets a uniqued IRValueType for the IR Type.
  IRValueType *getIRValueType(Type irType) {
    return getUniquedTypeNode<IRValueType>(irType);
  }

  /// Creates a new ObjectValueType.
  /// Object value types are not uniqued.
  ObjectValueType *
  newObjectValueType(ObjectValueType::IrTypeConstructor irCtor,
                     Identifier *typeIdentifier,
                     llvm::ArrayRef<Identifier *> fieldIdentifiers,
                     llvm::ArrayRef<TypeNode *> fieldTypes) {
    assert(fieldIdentifiers.size() == fieldTypes.size());
    size_t n = fieldIdentifiers.size();

    Identifier **allocFieldIdentifiers = allocator.Allocate<Identifier *>(n);
    std::copy_n(fieldIdentifiers.begin(), n, allocFieldIdentifiers);
    TypeNode **allocFieldTypes = allocator.Allocate<TypeNode *>(n);
    std::copy_n(fieldTypes.begin(), n, allocFieldTypes);
    auto *ovt = allocator.Allocate<ObjectValueType>(1);
    new (ovt) ObjectValueType(irCtor, typeIdentifier, n, allocFieldIdentifiers,
                              allocFieldTypes);
    return ovt;
  }

  /// Gets a CastType.
  CastType *getCastType(Identifier *typeIdentifier, TypeVar *typeVar) {
    return getUniquedTypeNode<CastType>(typeIdentifier, typeVar);
  }

  /// Gets a ReadType.
  ReadType *getReadType(TypeNode *type) {
    return getUniquedTypeNode<ReadType>(type);
  }

  /// Gets a WriteType.
  WriteType *getWriteType(TypeNode *type) {
    return getUniquedTypeNode<WriteType>(type);
  }

  /// Creates a Constraint.
  Constraint *getConstraint(TypeNode *t1, TypeNode *t2) {
    // Lookup based on a stack allocated key.
    Constraint v(t1, t2);
    auto it = constraintUniquer.insert(&v);
    if (!it.second)
      return *it.first;

    auto *av = allocator.Allocate<Constraint>(1);
    new (av) Constraint(v); // Copy ctor
    *it.first = av;         // Replace key pointer with durable allocation.
    addConstraintToGraph(av);
    currentEnvironment->getConstraints().insert(av);
    return av;
  }

  /// Creates a new ConstraintSet.
  ConstraintSet *newConstraintSet() {
    auto *cs = allocator.Allocate<ConstraintSet>(1);
    new (cs) ConstraintSet();
    return cs;
  }

  /// Creates a new TypeVarSet.
  TypeVarSet *newTypeVarSet() {
    auto *tvs = allocator.Allocate<TypeVarSet>(1);
    new (tvs) TypeVarSet();
    return tvs;
  }

  /// Gets a reference to the current members.
  /// This is the actual backing set. Any modification to the graph can
  /// invalidate iterators.
  const ValueTypeSet &getMembers(TypeNode *node) {
    return typeNodeMembers[node];
  }

private:
  /// Generically creates a uniquable TypeNode subclass.
  template <typename ConcreteTy, typename... Args>
  ConcreteTy *getUniquedTypeNode(Args &&... args) {
    // Lookup based on stack allocated key.
    ConcreteTy v(std::forward<Args>(args)...);
    auto it = typeUniquer.insert(&v);
    if (!it.second) {
      return static_cast<ConcreteTy *>(*it.first);
    }

    auto *av = allocator.Allocate<ConcreteTy>(1);
    new (av) ConcreteTy(v); // Copy ctor
    *it.first = av;         // Replace key pointer with durable allocation.
    return av;
  }

  /// Adds a constraint to the graph structure.
  void addConstraintToGraph(Constraint *c);

  /// Propagates any pending constraints.
  void propagateConstraints();

  // Configuration.
  IrTypeMapHook irTypeMapHook;

  // Allocation/uniquing management.
  llvm::BumpPtrAllocator allocator;
  llvm::DenseMap<StringRef, Identifier *> identifierMap;
  llvm::DenseSet<TypeNode *, TypeNode::PtrInfo> typeUniquer;
  llvm::DenseSet<Constraint *, Constraint::PtrInfo> constraintUniquer;
  int typeVarCounter = 0;

  // Graph management.
  llvm::DenseMap<TypeNode *, ConstraintSet> fwdNodeToConstraintMap;
  llvm::DenseMap<Constraint *, TypeNodeSet> fwdConstraintToNodeMap;
  llvm::DenseMap<TypeNode *, ConstraintSet> bakNodeToConstraintMap;
  // Note that we track contents for all TypeNodes, not just vars, as this
  // can be used to determine illegal dataflows.
  llvm::DenseMap<TypeNode *, ValueTypeSet> typeNodeMembers;

  // Propagation worklist.
  /// Constraints that are pending propagation.
  ConstraintSet pendingConstraints;
  ConstraintSet pendingConstraintWorklist;

  // Environment management.
  std::vector<std::unique_ptr<Environment>> environmentStack;
  Environment *currentEnvironment;
};

inline bool TypeNode::operator==(const TypeNode &that) const {
  if (getKind() != that.getKind())
    return false;
  switch (getKind()) {
  case Kind::TypeVar: {
    auto &thisCast = static_cast<const TypeVar &>(*this);
    auto &thatCast = static_cast<const TypeVar &>(that);
    return thisCast.getOrdinal() == thatCast.getOrdinal();
  }
  case Kind::CastType: {
    auto &thisCast = static_cast<const CastType &>(*this);
    auto &thatCast = static_cast<const CastType &>(that);
    return thisCast.getTypeIdentifier() == thatCast.getTypeIdentifier() &&
           thisCast.getTypeVar() == thatCast.getTypeVar();
  }
  case Kind::ReadType: {
    auto &thisCast = static_cast<const ReadType &>(*this);
    auto &thatCast = static_cast<const ReadType &>(that);
    return thisCast.getType() == thatCast.getType();
  }
  case Kind::WriteType: {
    auto &thisCast = static_cast<const WriteType &>(*this);
    auto &thatCast = static_cast<const WriteType &>(that);
    return thisCast.getType() == thatCast.getType();
  }
  case Kind::IRValueType: {
    auto &thisCast = static_cast<const IRValueType &>(*this);
    auto &thatCast = static_cast<const IRValueType &>(that);
    return thisCast.getIrType() == thatCast.getIrType();
  }
  case Kind::ObjectValueType:
    llvm_unreachable("ObjectValueType not implemented");
  default:
    llvm_unreachable("unhandled TypeNode subclass");
  }
  return false;
}

} // namespace CPA
} // namespace Typing
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_TYPING_CPA_SUPPORT_H
