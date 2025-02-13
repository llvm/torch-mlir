#pragma once

/// (almost) STL-compatible container that implements an associative map
/// iteratable according to insertion order. Mimicks Python Dict. Rationale: to
/// ease testing of the C++ importer against Python onnx_importer we need to
/// compare text outputs. MLIR values corresponding to tensors might be written
/// in different (compatible) orders due to differences in iteration order
/// between C++ STL unordered_map and Python Dict. Therefore we adopt the
/// insertion order here as well.

#include <unordered_map>
#include <vector>

namespace torch_mlir_onnx {

template <typename _Key, typename _Tp> struct DictIterator {
private:
  using key_type = _Key;
  using mapped_type = _Tp;
  using self = DictIterator<key_type, mapped_type>;
  using vector = std::vector<key_type>;
  using key_value_map = std::unordered_map<key_type, mapped_type>;
  using vector_iterator = typename vector::iterator;

  vector_iterator v_it_;
  key_value_map *m_ = nullptr;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = std::pair<const key_type, mapped_type>;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  DictIterator() = default;

  explicit DictIterator(const vector_iterator &it, key_value_map *m) noexcept
      : v_it_(it), m_(m) {}

  reference operator*() const noexcept { return *m_->find(*v_it_); }

  pointer operator->() const noexcept { return m_->find(*v_it_).operator->(); }

  self &operator++() noexcept {
    ++v_it_;
    return *this;
  }

  self operator++(int) noexcept {
    self _tmp(*this);
    ++*this;
    return _tmp;
  }

  friend bool operator==(const self &x, const self &y) noexcept {
    return x.v_it_ == y.v_it_ && x.m_ == y.m_;
  }
};

template <typename _Key, typename _Tp> class DictConstIterator {
private:
  using key_type = _Key;
  using mapped_type = _Tp;
  using self = DictConstIterator<key_type, mapped_type>;
  using vector = std::vector<key_type>;
  using key_value_map = std::unordered_map<key_type, mapped_type>;
  using vector_const_iterator = typename vector::const_iterator;

  vector_const_iterator v_it_;
  const key_value_map *m_ = nullptr;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = std::pair<const key_type, mapped_type>;
  using difference_type = std::ptrdiff_t;
  using pointer = const value_type *;
  using reference = const value_type &;

  DictConstIterator() = default;

  explicit DictConstIterator(const vector_const_iterator &it,
                             const key_value_map *m) noexcept
      : v_it_(it), m_(m) {}

  reference operator*() const noexcept { return *m_->find(*v_it_); }

  pointer operator->() const noexcept { return m_->find(*v_it_).operator->(); }

  self &operator++() noexcept {
    ++v_it_;
    return *this;
  }

  self operator++(int) noexcept {
    self _tmp(*this);
    ++*this;
    return _tmp;
  }

  friend bool operator==(const self &x, const self &y) noexcept {
    return x.v_it_ == y.v_it_ && x.m_ == y.m_;
  }
};

template <typename _Key, typename _Tp> class Dict {

private:
  using key_value_map = std::unordered_map<_Key, _Tp>;
  using key_vector = std::vector<_Key>;
  using key_index_map =
      std::unordered_map<_Key, typename key_vector::iterator::difference_type>;

  key_value_map m_;
  key_vector k_;
  key_index_map i_;

public:
  /// Public typedefs.
  using key_type = _Key;
  using mapped_type = _Tp;
  using value_type = std::pair<const _Key, _Tp>;
  using size_type = std::size_t;
  using allocator_type = std::allocator<value_type>;

  ///  Iterator-related typedefs.
  using reference = mapped_type &;
  using const_reference = const mapped_type &;
  using pointer = typename std::allocator_traits<allocator_type>::pointer;
  using const_pointer =
      typename std::allocator_traits<allocator_type>::const_pointer;
  using iterator = DictIterator<key_type, mapped_type>;
  using const_iterator = DictConstIterator<key_type, mapped_type>;

  /* Constructors, assignment and destructor */
  Dict() = default;
  Dict(const Dict &) = default;
  Dict(Dict &&) = default;

  Dict &operator=(const Dict &) = default;
  Dict &operator=(Dict &&) = default;

  ~Dict() = default;

  /* Selectors */
  const_iterator find(const key_type &key) const {
    auto ii = i_.find(key);
    if (ii == i_.end())
      return end();
    return const_iterator{k_.cbegin() + (*ii).second, &m_};
  }
  size_type size() const { return m_.size(); }
  bool empty() const { return m_.empty(); }
  reference at(const key_type &key) { return m_.at(key); }
  const_reference at(const key_type &key) const { return m_.at(key); }

  /* Mutators */
  iterator find(const key_type &key) {
    auto ii = i_.find(key);
    if (ii == i_.end())
      return end();
    return iterator{k_.begin() + (*ii).second, &m_};
  }
  std::pair<iterator, bool> insert(const value_type &pair) {
    auto found_it = find(pair.first);
    if (found_it == end()) {
      auto ki = k_.insert(k_.end(), pair.first);
      i_.emplace(pair.first, ki - k_.begin());
      m_.insert(pair);
      return {iterator{ki, &m_}, true};
    }
    return {found_it, false};
  }
  std::pair<iterator, bool> insert(value_type &&pair) {
    auto found_it = find(pair.first);
    if (found_it == end()) {
      auto ki = k_.insert(k_.end(), pair.first);
      i_.emplace(pair.first, ki - k_.begin());
      m_.insert(std::move(pair));
      return {iterator{ki, &m_}, true};
    }
    return {found_it, false};
  }

  template <typename... _Args>
  std::pair<iterator, bool> emplace(_Args &&...args) {
    return insert(value_type(std::forward<_Args>(args)...));
  }
  reference operator[](const key_type &key) {
    auto ins = emplace(key, mapped_type());
    return (*ins.first).second;
  }

  /* Iterators */
  iterator begin() { return iterator(k_.begin(), &m_); }
  const_iterator begin() const { return const_iterator(k_.cbegin(), &m_); }
  const_iterator cbegin() const { return const_iterator(k_.cbegin(), &m_); }
  iterator end() { return iterator(k_.end(), &m_); }
  const_iterator end() const { return const_iterator(k_.cend(), &m_); }
  const_iterator cend() const { return const_iterator(k_.cend(), &m_); }
};

} // namespace torch_mlir_onnx
